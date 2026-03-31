
import threading
import queue
import time
from typing import Dict, Any, Callable


class TaskQueue:
    def __init__(self, max_workers=4):
        self.queue = queue.PriorityQueue()
        self.results = {}
        self.lock = threading.Lock()
        self.workers = []
        self.running = True
        for _ in range(max_workers):
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.workers.append(t)

    def submit(self, task_id: str, func: Callable, priority: int = 0, *args, **kwargs):
        self.queue.put((priority, task_id, func, args, kwargs))
        return task_id

    def _worker(self):
        while self.running:
            try:
                priority, task_id, func, args, kwargs = self.queue.get(
                    timeout=1)
                # 添加日志：任务开始
                print(f"[TaskQueue Worker] 开始执行任务 {task_id}")
                result = func(*args, **kwargs)
                with self.lock:
                    self.results[task_id] = result
                # 添加日志：任务完成
                print(f"[TaskQueue Worker] 任务 {task_id} 执行完成")
            except queue.Empty:
                continue
            except Exception as e:
                # 添加日志：任务异常
                print(f"[TaskQueue Worker] 任务 {task_id} 异常: {e}")
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.results[task_id] = {"error": str(e)}

    def get_result(self, task_id, timeout=None):
        start = time.time()
        while time.time() - start < (timeout or 60):
            with self.lock:
                if task_id in self.results:
                    return self.results.pop(task_id)
            time.sleep(0.1)
        return None

    def shutdown(self):
        self.running = False
        for t in self.workers:
            t.join(timeout=1)
