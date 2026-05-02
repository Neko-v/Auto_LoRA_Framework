# /core/process_runner.py

import os
import signal
import subprocess


NOISY_LOG_PATTERNS = [
    "Redirects are currently not supported in Windows or MacOs",
]

ACTIVE_PROCESS = {
    "process": None
}


def kill_process_tree(pid):
    """
    尽量杀掉整个子进程树
    Windows 下使用 taskkill /T /F
    """
    if pid is None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def cancel_active_process():
    """
    供 GUI 停止按钮调用
    返回 True 表示确实找到并终止了一个活跃子进程
    """
    process = ACTIVE_PROCESS.get("process")

    if process is None:
        return False

    if process.poll() is not None:
        return False

    kill_process_tree(process.pid)
    return True


def stream_subprocess(cmd, cwd=None, env=None):
    """
    流式执行子进程, 实时产出 stdout/stderr 日志
    支持在中断时清理整个子进程树
    """
    run_env = os.environ.copy()

    if env:
        run_env.update(env)

    run_env["PYTHONUNBUFFERED"] = "1"
    run_env["PYTHONIOENCODING"] = "utf-8"

    creationflags = 0

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )

    ACTIVE_PROCESS["process"] = process

    try:
        if process.stdout is not None:
            for raw_line in iter(process.stdout.readline, ""):
                line = raw_line.rstrip("\r\n")

                if not line:
                    continue

                if any(pattern in line for pattern in NOISY_LOG_PATTERNS):
                    continue

                yield line

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    except (KeyboardInterrupt, GeneratorExit):
        kill_process_tree(process.pid)
        raise

    except Exception:
        if process.poll() is None:
            kill_process_tree(process.pid)
        raise

    finally:
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass

        if ACTIVE_PROCESS.get("process") is process:
            ACTIVE_PROCESS["process"] = None