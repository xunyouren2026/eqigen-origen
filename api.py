from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import uuid
import os
import sys
from typing import Optional, List
import cv2
import numpy as np
import shutil
from task_queue import TaskQueue

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局脚本生成器实例（懒加载）
_script_gen_instance = None


def get_script_generator():
    """懒加载脚本生成器，根据配置初始化"""
    global _script_gen_instance
    if _script_gen_instance is None:
        if app.state.inferencer is None:
            raise RuntimeError("Inferencer not initialized")
        cfg = app.state.inferencer.config.script_gen
        if not cfg.enabled:
            raise HTTPException(status_code=501, detail="脚本生成功能已禁用，请检查配置。")

        from script_generator import ScriptGenerator
        if cfg.mode == "local":
            _script_gen_instance = ScriptGenerator(
                use_local=True,
                model_name=cfg.model_name,
                api_key=None
            )
        elif cfg.mode == "api":
            if not cfg.api_key:
                raise HTTPException(
                    status_code=500, detail="API 模式需要提供 api_key")
            _script_gen_instance = ScriptGenerator(
                use_local=False,
                model_name=cfg.model_name,
                api_key=cfg.api_key
            )
        else:
            raise HTTPException(
                status_code=500, detail=f"未知的脚本生成模式: {cfg.mode}")
    return _script_gen_instance


# 主界面 HTML 文件路径（请确保此文件存在于项目根目录）
HTML_FILE_PATH = "webui.html"


@app.on_event("startup")
async def startup_event():
    if not hasattr(app.state, 'inferencer'):
        app.state.inferencer = None
    if not hasattr(app.state, 'results'):
        app.state.results = {}
    if not hasattr(app.state, 'tasks'):
        app.state.tasks = {}
    if not hasattr(app.state, 'task_queue'):
        app.state.task_queue = TaskQueue(max_workers=4)
    print(f"API 启动完成，状态已初始化，任务队列工作线程数: {len(app.state.task_queue.workers)}")


@app.get("/")
async def serve_ui():
    """提供主界面 HTML"""
    if not os.path.exists(HTML_FILE_PATH):
        return HTMLResponse("<h1>界面文件未找到</h1><p>请确保 webui.html 存在于项目根目录。</p>", status_code=404)
    with open(HTML_FILE_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ========== 提取生成任务函数（供单个和批量共用） ==========
def run_generation_task(task_id, inferencer, args):
    """执行单个生成任务（被任务队列调用）"""
    print(f"[任务 {task_id}] 开始执行...")
    try:
        print(f"[任务 {task_id}] 调用 inferencer.generate，参数: prompt={args.get('prompt')}, duration={args.get('duration')}, fps={args.get('fps')}, steps={args.get('num_steps')}, resolution={args.get('resolution')}")
        sys.stdout.flush()

        result_path = inferencer.generate(**args)
        print(f"[任务 {task_id}] inferencer.generate 完成，结果路径: {result_path}")

        # 移动结果文件到最终路径
        shutil.move(result_path, args['output_path'])
        app.state.results[task_id] = args['output_path']
        print(f"[任务 {task_id}] 结果已保存到: {args['output_path']}")

        # 清理临时文件
        # 修复：处理 audio_paths 可能为 None 的情况
        audio_paths = args.get('audio_paths') or []
        for p in audio_paths:
            try:
                os.remove(p)
                print(f"[任务 {task_id}] 清理临时音频文件: {p}")
            except Exception as e:
                print(f"[任务 {task_id}] 清理临时音频文件失败: {e}")
        if args.get('bgm_file_path') and os.path.exists(args['bgm_file_path']):
            try:
                os.remove(args['bgm_file_path'])
                print(f"[任务 {task_id}] 清理临时背景音乐文件: {args['bgm_file_path']}")
            except Exception as e:
                print(f"[任务 {task_id}] 清理临时背景音乐文件失败: {e}")

        # 从任务列表中移除（标记完成）
        if task_id in app.state.tasks:
            del app.state.tasks[task_id]
        print(f"[任务 {task_id}] 执行完成，已从任务列表中移除")

    except Exception as e:
        print(f"[任务 {task_id}] 错误 - {str(e)}")
        import traceback
        traceback.print_exc()
        app.state.results[task_id] = {"error": str(e)}
        if task_id in app.state.tasks:
            app.state.tasks[task_id] = {
                "status": "error", "detail": str(e)}


@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    negative: str = Form(""),
    duration: float = Form(2.0),
    fps: int = Form(8),
    cfg_scale: float = Form(7.5),
    steps: int = Form(50),
    style: str = Form(""),
    camera: str = Form(""),
    output_format: str = Form("mp4"),
    watermark: str = Form(""),
    resolution: str = Form("256p"),
    async_mode: bool = Form(False),
    # 高级选项
    ar_mode: bool = Form(False),
    use_memory: bool = Form(True),
    distill_mode: bool = Form(False),
    use_raft: bool = Form(False),
    use_routing: bool = Form(False),
    use_compression: bool = Form(False),
    use_pyramid: bool = Form(False),
    use_learned_compressor: bool = Form(False),
    use_concatenated_history: bool = Form(False),
    temporal_smooth: bool = Form(False),
    interpolate: bool = Form(False),
    superres: bool = Form(False),
    # 新增加速选项
    use_parallel_tile: bool = Form(False),
    tile_batch_size: int = Form(4),
    use_tensorrt: bool = Form(False),
    tensorrt_engine_path: str = Form("model.trt"),
    physics_correct: bool = Form(False),
    use_pipeline: bool = Form(False),
    # 背景音乐
    bgm_url: str = Form(""),
    bgm_file: Optional[UploadFile] = File(None),
    # 多模态文件
    init_image: Optional[UploadFile] = File(None),
    reference_images: Optional[List[UploadFile]] = File(None),
    reference_videos: Optional[List[UploadFile]] = File(None),
    audio_files: Optional[List[UploadFile]] = File(None),
    lens_script: Optional[UploadFile] = File(None)
):
    if app.state.inferencer is None:
        raise HTTPException(
            status_code=500, detail="Inferencer not initialized")

    task_id = str(uuid.uuid4())
    print(f"收到请求 - 任务ID: {task_id}, prompt: {prompt}")

    temp_dir = app.state.inferencer.config.api.temp_dir
    output_path = os.path.join(temp_dir, f"{task_id}.{output_format}")

    # 处理初始图像
    init_image_np = None
    if init_image and init_image.filename and init_image.filename.strip():
        try:
            content = init_image.file.read()
            if content:
                nparr = np.frombuffer(content, np.uint8)
                init_image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if init_image_np is not None:
                    init_image_np = cv2.cvtColor(
                        init_image_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"处理初始图像错误: {e}")

    # 参考图片
    ref_imgs = []
    if reference_images:
        for img_file in reference_images:
            try:
                content = img_file.file.read()
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ref_imgs.append(img)
            except Exception as e:
                print(f"处理参考图像错误: {e}")

    # 参考视频
    ref_vids = []
    if reference_videos:
        for vid_file in reference_videos:
            try:
                content = vid_file.file.read()
                tmp_path = os.path.join(
                    temp_dir, f"{task_id}_tmp_vid_{len(ref_vids)}.mp4")
                with open(tmp_path, "wb") as f:
                    f.write(content)
                cap = cv2.VideoCapture(tmp_path)
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if frames:
                    ref_vids.append(np.array(frames))
                os.remove(tmp_path)
            except Exception as e:
                print(f"处理参考视频错误: {e}")

    # 音频文件
    audio_paths = []
    if audio_files:
        for aud_file in audio_files:
            tmp_path = os.path.join(
                temp_dir, f"{task_id}_tmp_aud_{len(audio_paths)}.wav")
            with open(tmp_path, "wb") as f:
                f.write(aud_file.file.read())
            audio_paths.append(tmp_path)

    # 镜头脚本
    lens_path = None
    if lens_script and lens_script.filename and lens_script.filename.strip():
        lens_path = os.path.join(temp_dir, f"{task_id}_lens.json")
        with open(lens_path, "wb") as f:
            f.write(lens_script.file.read())

    # 背景音乐文件临时保存
    bgm_file_path = None
    if bgm_file and bgm_file.filename and bgm_file.filename.strip():
        bgm_file_path = os.path.join(
            temp_dir, f"{task_id}_bgm{os.path.splitext(bgm_file.filename)[1]}")
        with open(bgm_file_path, "wb") as f:
            f.write(bgm_file.file.read())

    # 组装参数
    args = {
        'prompt': prompt,
        'negative_prompt': negative,
        'duration': duration,
        'fps': fps,
        'cfg_scale': cfg_scale,
        'num_steps': steps,
        'style': style if style != "无" else None,
        'camera': camera if camera != "无" else None,
        'output_format': output_format,
        'watermark': watermark,
        'init_image': init_image_np,
        'reference_images': ref_imgs if ref_imgs else None,
        'reference_videos': ref_vids if ref_vids else None,
        'audio_paths': audio_paths if audio_paths else None,
        'lens_script_path': lens_path,
        'resolution': resolution,
        'ar_mode': ar_mode,
        'use_memory': use_memory,
        'distill_mode': distill_mode,
        'use_raft': use_raft,
        'use_routing': use_routing,
        'use_compression': use_compression,
        'use_pyramid': use_pyramid,
        'use_learned_compressor': use_learned_compressor,
        'use_concatenated_history': use_concatenated_history,
        'temporal_smooth': temporal_smooth,
        'interpolate': interpolate,
        'superres': superres,
        'use_parallel_tile': use_parallel_tile,
        'tile_batch_size': tile_batch_size,
        'use_tensorrt': use_tensorrt,
        'tensorrt_engine_path': tensorrt_engine_path,
        'physics_correct': physics_correct,
        'use_pipeline': use_pipeline,
        'bgm_url': bgm_url,
        'bgm_file_path': bgm_file_path,
        'output_path': output_path,
        'async_mode': async_mode,
    }

    if async_mode:
        app.state.tasks[task_id] = {"status": "pending", "progress": 0}
        # 修正：将 task_id 作为位置参数传给 run_generation_task
        app.state.task_queue.submit(
            task_id, run_generation_task, 0, task_id, app.state.inferencer, args)
        print(f"任务 {task_id}: 已添加到队列")
        return JSONResponse({"task_id": task_id, "status": "pending"})
    else:
        # 同步执行（实际中很少用）
        run_generation_task(task_id, app.state.inferencer, args)
        if isinstance(app.state.results[task_id], dict) and "error" in app.state.results[task_id]:
            raise HTTPException(
                status_code=500, detail=app.state.results[task_id]["error"])
        return FileResponse(output_path, media_type=f"video/{output_format}")


# ========== 新增批量生成端点 ==========
@app.post("/generate_batch")
async def generate_batch(
    prompts: List[str] = Form(...),
    negative: str = Form(""),
    duration: float = Form(2.0),
    fps: int = Form(8),
    cfg_scale: float = Form(7.5),
    steps: int = Form(50),
    style: str = Form(""),
    camera: str = Form(""),
    output_format: str = Form("mp4"),
    watermark: str = Form(""),
    resolution: str = Form("256p"),
    # 其他参数（与单个生成相同，但暂不支持文件上传）
    ar_mode: bool = Form(False),
    use_memory: bool = Form(True),
    distill_mode: bool = Form(False),
    use_raft: bool = Form(False),
    use_routing: bool = Form(False),
    use_compression: bool = Form(False),
    use_pyramid: bool = Form(False),
    use_learned_compressor: bool = Form(False),
    use_concatenated_history: bool = Form(False),
    temporal_smooth: bool = Form(False),
    interpolate: bool = Form(False),
    superres: bool = Form(False),
    use_parallel_tile: bool = Form(False),
    tile_batch_size: int = Form(4),
    use_tensorrt: bool = Form(False),
    tensorrt_engine_path: str = Form("model.trt"),
    physics_correct: bool = Form(False),
    use_pipeline: bool = Form(False),
    bgm_url: str = Form(""),
    # 注意：批量生成暂不支持多模态文件（如参考图像、视频等），可扩展
):
    if app.state.inferencer is None:
        raise HTTPException(
            status_code=500, detail="Inferencer not initialized")

    task_ids = []
    temp_dir = app.state.inferencer.config.api.temp_dir
    for prompt in prompts:
        task_id = str(uuid.uuid4())
        output_path = os.path.join(temp_dir, f"{task_id}.{output_format}")
        args = {
            'prompt': prompt,
            'negative_prompt': negative,
            'duration': duration,
            'fps': fps,
            'cfg_scale': cfg_scale,
            'num_steps': steps,
            'style': style if style != "无" else None,
            'camera': camera if camera != "无" else None,
            'output_format': output_format,
            'watermark': watermark,
            'resolution': resolution,
            'ar_mode': ar_mode,
            'use_memory': use_memory,
            'distill_mode': distill_mode,
            'use_raft': use_raft,
            'use_routing': use_routing,
            'use_compression': use_compression,
            'use_pyramid': use_pyramid,
            'use_learned_compressor': use_learned_compressor,
            'use_concatenated_history': use_concatenated_history,
            'temporal_smooth': temporal_smooth,
            'interpolate': interpolate,
            'superres': superres,
            'use_parallel_tile': use_parallel_tile,
            'tile_batch_size': tile_batch_size,
            'use_tensorrt': use_tensorrt,
            'tensorrt_engine_path': tensorrt_engine_path,
            'physics_correct': physics_correct,
            'use_pipeline': use_pipeline,
            'bgm_url': bgm_url,
            'bgm_file_path': None,  # 批量暂不支持背景音乐文件
            'init_image': None,
            'reference_images': None,
            'reference_videos': None,
            'audio_paths': None,
            'lens_script_path': None,
            'output_path': output_path,
            'async_mode': True,
        }
        app.state.tasks[task_id] = {"status": "pending", "progress": 0}
        # 修正：将 task_id 作为位置参数传给 run_generation_task
        app.state.task_queue.submit(
            task_id, run_generation_task, 0, task_id, app.state.inferencer, args)
        task_ids.append(task_id)

    return JSONResponse({"task_ids": task_ids, "status": "pending"})


# 其他端点保持不变
@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id in app.state.results:
        result = app.state.results.pop(task_id)
        if isinstance(result, dict) and "error" in result:
            return JSONResponse({"status": "error", "detail": result["error"]}, status_code=500)
        else:
            return FileResponse(result, media_type="video/mp4")
    else:
        if task_id in app.state.tasks:
            return JSONResponse({"status": "pending", "task_id": task_id})
        else:
            return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    if task_id in app.state.tasks:
        return JSONResponse(app.state.tasks[task_id])
    else:
        return JSONResponse({"status": "not_found"}, status_code=404)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "inferencer_ready": app.state.inferencer is not None,
        "tasks_count": len(app.state.tasks),
        "results_count": len(app.state.results)
    }


# ========== 故事生成脚本端点（懒加载） ==========
@app.post("/generate_script")
async def generate_script(
    story: str = Form(..., description="故事创意文本")
):
    """
    根据故事创意生成镜头脚本（JSON格式）
    """
    if not story or not story.strip():
        raise HTTPException(status_code=400, detail="故事创意不能为空")
    try:
        gen = get_script_generator()
        script = gen.generate(story)
        return JSONResponse(script)
    except Exception as e:
        print(f"脚本生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"脚本生成失败: {str(e)}")
