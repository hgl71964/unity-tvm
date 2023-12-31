import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.cost_model import PyCostModel, RandomModel, XGBModel
from tvm.meta_schedule.testing.dummy_object import DummyBuilder, DummyRunner
from tvm.script import tir as T
from tvm.tir import Schedule

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# see `tests/python/unittest/test_meta_schedule_task_scheduler.py` for more
print("""
# ###########
# tvm module
# ###########
""")


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulReluModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        d: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        D = T.match_buffer(d, (1024, 1024), "float32")
        C = T.alloc_buffer((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)  # type: ignore


@tvm.script.ir_module
class BatchMatmulModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 128, 128])
        B = T.match_buffer(b, [16, 128, 128])
        C = T.match_buffer(c, [16, 128, 128])
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("matmul"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    C[vn, vi, vj] = 0.0  # type: ignore
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


print("""
# ###########
# schedule
# ###########
""")


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_batch_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k, t = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 2, 2, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[2, 4, 64, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    t_0, t_1 = sch.split(loop=t, factors=[2, 512])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, t_0, t_1)


print("""
# ###########
# run
# ###########
""")


def test_auto_space_generator():
    num_trials_per_iter = 3
    max_trials_per_task = 10
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    round_robin.tune(
        [
            ms.TuneContext(
                MatmulModule,
                # target=tvm.target.Target("llvm"),
                target=tvm.target.Target("llvm -num-cores 4"),
                space_generator="post-order-apply",  # will call `create`
                search_strategy="evolutionary",  # will call `create`
                task_name="Test",
                rand_state=42,
                logger=logging.getLogger().setLevel(logging.INFO),
            )
        ],
        [1.0],
        max_trials_global=num_trials_per_iter,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=64,
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        cost_model=XGBModel(),
    )
    assert len(database) == max_trials_per_task


def test_meta_schedule_task_scheduler_single():
    num_trials_per_iter = 3
    max_trials_per_task = 10
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    round_robin.tune(
        [
            ms.TuneContext(
                MatmulModule,
                target=tvm.target.Target("llvm"),
                space_generator=_schedule_matmul,
                search_strategy=ms.search_strategy.ReplayTrace(),
                task_name="Test",
                rand_state=42,
            )
        ],
        [1.0],
        max_trials_global=num_trials_per_iter,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=64,
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        cost_model=None,
    )
    assert len(database) == max_trials_per_task


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 6
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=_schedule_batch_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    round_robin.tune(
        tasks,
        [1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials_global=max_trials_per_task * len(tasks),
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        cost_model=None,
    )
    # assert len(database) == max_trials_per_task * len(tasks)
    # for task in tasks:
    #     assert (
    #         len(
    #             database.get_top_k(
    #                 database.commit_workload(task.mod),
    #                 100000,
    #             )
    #         )
    #         == max_trials_per_task
    #     )


test_auto_space_generator()
# test_meta_schedule_task_scheduler_single()
# test_meta_schedule_task_scheduler_multiple()
