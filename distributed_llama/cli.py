"""CLI entry point for distributed-llama-python.

Port of src/dllama.cpp main() and src/app.cpp argument parsing.

Usage:
    dllama inference --model <path> --tokenizer <path> --prompt "..."
    dllama chat --model <path> --tokenizer <path>
    dllama perplexity --model <path> --tokenizer <path> --prompt "..."
    dllama worker --host <addr> --port <port>
    dllama-api --model <path> --tokenizer <path>
"""

import sys
import argparse
from .quants import F_32, F_16, F_Q40, F_Q80
from .tokenizer import ChatTemplateType


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dllama",
        description="Distributed LLM inference — Python port",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common_args(p):
        p.add_argument("--model", required=True, help="Path to model file")
        p.add_argument("--tokenizer", required=True, help="Path to tokenizer file")
        p.add_argument("--buffer-float-type", default="f32",
                       choices=["f32", "f16", "q40", "q80"],
                       help="Float precision for sync buffers")
        p.add_argument("--workers", nargs="*", default=[],
                       help="Worker addresses (ip:port)")
        p.add_argument("--max-seq-len", type=int, default=0,
                       help="Maximum sequence length")
        p.add_argument("--nthreads", type=int, default=1,
                       help="Number of threads (FIXME: >1 causes issues)")
        p.add_argument("--nbatches", type=int, default=32,
                       help="Batch size for eval phase")
        p.add_argument("--seed", type=int, default=0,
                       help="Random seed for sampling")
        p.add_argument("--turbo", action="store_true",
                       help="Enable TCP_NODELAY")

    # inference
    p_inf = sub.add_parser("inference", help="Run inference with benchmark")
    add_common_args(p_inf)
    p_inf.add_argument("--prompt", required=True, help="Input prompt")
    p_inf.add_argument("--steps", type=int, default=256,
                       help="Number of tokens to generate")
    p_inf.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0=greedy)")
    p_inf.add_argument("--topp", type=float, default=0.9,
                       help="Top-p sampling threshold")

    # chat
    p_chat = sub.add_parser("chat", help="Run interactive chat")
    add_common_args(p_chat)
    p_chat.add_argument("--temperature", type=float, default=0.0)
    p_chat.add_argument("--topp", type=float, default=0.9)
    p_chat.add_argument("--chat-template", type=str, default=None,
                        choices=["llama2", "llama3", "deepseek3", "chatml"],
                        help="Force chat template type")

    # perplexity
    p_perp = sub.add_parser("perplexity", help="Compute perplexity")
    add_common_args(p_perp)
    p_perp.add_argument("--prompt", required=True, help="Input prompt")

    # worker
    p_work = sub.add_parser("worker", help="Run worker node")
    p_work.add_argument("--host", default="0.0.0.0", help="Bind address for worker")
    p_work.add_argument("--port", type=int, default=9999, help="Port to listen on")
    p_work.add_argument("--nthreads", type=int, default=4, help="Number of threads")

    # api
    p_api = sub.add_parser("api", help="Run API server")
    add_common_args(p_api)
    p_api.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_api.add_argument("--port", type=int, default=8080, help="Bind port")

    return parser


def parse_workers(worker_strs):
    """Parse worker addresses like '10.0.0.2:9999'."""
    hosts = []
    ports = []
    for ws in worker_strs:
        if ":" in ws:
            host, port = ws.rsplit(":", 1)
            hosts.append(host)
            ports.append(int(port))
        else:
            hosts.append(ws)
            ports.append(9999)
    return hosts, ports


def main():
    parser = create_parser()
    args = parser.parse_args()

    from .inference import (
        AppCliArgs, run_inference_app, run_worker_app,
        inference_mode, perplexity_mode, chat_mode,
    )

    cli_args = AppCliArgs()
    cli_args.mode = args.mode

    sync_type_map = {"f32": F_32, "f16": F_16, "q40": F_Q40, "q80": F_Q80}
    cli_args.sync_type = sync_type_map.get(
        getattr(args, "buffer_float_type", "f32"), F_32,
    )

    template_map = {
        "llama2": ChatTemplateType.LLAMA2,
        "llama3": ChatTemplateType.LLAMA3,
        "deepseek3": ChatTemplateType.DEEPSEEK3,
        "chatml": ChatTemplateType.CHATML,
    }
    cli_args.chat_template_type = template_map.get(
        getattr(args, "chat_template", None) or "",
        ChatTemplateType.UNKNOWN,
    )

    if args.mode == "worker":
        cli_args.host = args.host
        cli_args.port = args.port
        cli_args.n_threads = args.nthreads
        run_worker_app(cli_args)
        return

    # Root node setup
    worker_hosts, worker_ports = parse_workers(
        getattr(args, "workers", []) or [],
    )
    cli_args.model_path = args.model
    cli_args.tokenizer_path = args.tokenizer
    cli_args.n_threads = args.nthreads
    cli_args.n_batches = getattr(args, "nbatches", 8)
    cli_args.max_seq_len = getattr(args, "max_seq_len", 0)
    cli_args.seed = getattr(args, "seed", 0)
    cli_args.net_turbo = getattr(args, "turbo", False)
    cli_args.n_workers = len(worker_hosts)
    cli_args.worker_hosts = worker_hosts
    cli_args.worker_ports = worker_ports

    if args.mode == "inference":
        cli_args.prompt = args.prompt
        cli_args.steps = args.steps
        cli_args.temperature = args.temperature
        cli_args.topp = args.topp
        cli_args.benchmark = True
        run_inference_app(cli_args, inference_mode)

    elif args.mode == "chat":
        cli_args.temperature = args.temperature
        cli_args.topp = args.topp
        run_inference_app(cli_args, chat_mode)

    elif args.mode == "perplexity":
        cli_args.prompt = args.prompt
        run_inference_app(cli_args, perplexity_mode)

    elif args.mode == "api":
        cli_args.host = args.host
        cli_args.port = args.port
        cli_args.temperature = getattr(args, "temperature", 0.0)
        cli_args.topp = getattr(args, "topp", 0.9)
        from .api import run_api_server
        run_api_server(cli_args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
