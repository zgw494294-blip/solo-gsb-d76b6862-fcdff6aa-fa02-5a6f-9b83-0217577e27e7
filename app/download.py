"""已提交批次 -> gzip NDJSON 的流式下载。

内存有界：逐页（PAGE_ROWS 行）从主库按键集顺序读取记录，逐行序列化并喂入
zlib gzip 压缩器；任何时刻内存里只有一小页记录与压缩器/一行 JSON 字节，
不缓存整批 NDJSON、解压正文或完整压缩文件，也不生成临时文件。

响应是单个完整 gzip 流：gzip 头 + 压缩数据 + CRC32/ISIZE 尾部（由 zlib
compressobj(wbits=31) 的 flush() 生成）。读路径只执行 SELECT，不修改批次。

SQLite 查询与 zlib 压缩放入线程执行，事件循环只负责向慢客户端背压式发送；
WAL 下读不持主库写锁，慢下载不会阻塞其他批次的上传提交或已完成批次的读取。
正常结束与客户端中途断开（任务被取消）都会关闭压缩器、游标与连接。
"""
from __future__ import annotations

import json
import sqlite3
import zlib
from typing import AsyncIterator

from anyio import CancelScope, to_thread

from . import db

# 每一小页读取的行数：最坏情况下内存只保留该页行 + 其 JSON 压缩块
PAGE_ROWS = 64
# gzip 压缩级别（gzip 命令默认即 6）；mtime=0 使头部确定
GZIP_LEVEL = 6


def _serialize_row(row: sqlite3.Row) -> bytes:
    """重新序列化一行：键固定为 id、message，值与库内原文逐字符一致。

    ensure_ascii=False 输出 UTF-8 原文码点（非 ASCII 不转义）；其余按 JSON
    规范转义（引号、反斜杠、换行/制表符等控制字符均以反斜杠转义），
    行尾固定一个 \\n，包括末行。
    """
    return (
        json.dumps(
            {"id": row["id"], "message": row["message"]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _compress_page(compressor, rows: list[sqlite3.Row]) -> bytes:
    """把一页 NDJSON 行压缩为一个 gzip 数据块（不结束 gzip 流）。"""
    return compressor.compress(b"".join(_serialize_row(row) for row in rows))


async def gzip_batch_stream(batch_id: str) -> AsyncIterator[bytes]:
    """生成该批次完整 gzip 流的异步块序列；调用前须已确认批次存在。

    连接与压缩器均在生成器内部创建，正常结束或 GeneratorExit
    （客户端中途断开导致响应任务被取消）都会在 finally 中释放，
    取消场景下用 shield 作用域保证清理本身不被再次取消。
    """
    conn = await to_thread.run_sync(db.connect)
    compressor = zlib.compressobj(
        GZIP_LEVEL, zlib.DEFLATED, 31  # wbits=31：gzip 封装（含 CRC/ISIZE 尾部）
    )
    try:
        after_seq = 0
        while True:
            rows = await to_thread.run_sync(
                db.get_lines_page, conn, batch_id, after_seq, PAGE_ROWS
            )
            if not rows:
                break
            chunk = await to_thread.run_sync(_compress_page, compressor, rows)
            if chunk:
                yield chunk
            after_seq = rows[-1]["seq"]
            if len(rows) < PAGE_ROWS:
                break
        # 结束 deflate 流并写出 gzip CRC32 + ISIZE 尾部
        trailer = await to_thread.run_sync(compressor.flush)
        if trailer:
            yield trailer
    finally:
        def _cleanup() -> None:
            try:
                conn.close()  # 同时释放该连接上的所有游标/语句
            except sqlite3.Error:
                pass

        with CancelScope(shield=True):
            await to_thread.run_sync(_cleanup)
