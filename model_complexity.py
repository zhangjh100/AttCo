"""Inference complexity, with fvcore's convention: one multiply-add = one FLOP."""
import argparse
import json
from pathlib import Path

import torch


def input_shape(layout, image_size=(224, 224), volume_size=(32, 128, 128)):
    if layout == '3d':
        return (1, 1, *volume_size)
    if layout == '2.5d':
        return (1, 3, *image_size)
    if layout == '2d':
        return (1, 1, *image_size)
    raise ValueError(f'Unknown input layout: {layout}')


def measure_complexity(model, shape, calculate_flops=True):
    """Count all parameters and trace one eval forward; restore module modes.

    Unsupported arithmetic is reported separately: counted FLOPs are a partial
    estimate, not a complete total. No backward/optimizer operations are counted.
    """
    result = {
        'params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_shape': list(shape), 'batch_size': shape[0],
        'mode': 'eval', 'backend': 'fvcore',
        'flop_convention': 'one multiply-add = one FLOP',
        'gflops': None, 'counted_gflops': None, 'status': 'disabled',
    }
    result['params_m'] = result['params'] / 1e6
    if not calculate_flops:
        return result
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        result.update(status='unavailable', error='Install fvcore: pip install fvcore')
        return result
    modes = [(module, module.training) for module in model.modules()]
    # MedNeXt-derived 3D models keep checkpointing enabled even in eval.
    # Bypass its autograd functions during JIT tracing, then restore the flags.
    checkpoint_flags = [(module, name, getattr(module, name))
                        for module in model.modules()
                        for name in ('outside_block_checkpointing',
                                     'inside_block_checkpointing')
                        if isinstance(getattr(module, name, None), bool)]
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device('cpu')
    dtype = parameter.dtype if parameter is not None else torch.float32
    try:
        model.eval()
        for module, name, _ in checkpoint_flags:
            setattr(module, name, False)
        with torch.no_grad(), torch.random.fork_rng(
                devices=[device.index] if device.type == 'cuda' else []):
            sample = torch.zeros(shape, device=device, dtype=dtype)
            analysis = FlopCountAnalysis(model, (sample,))
            analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            total = float(analysis.total())
            unsupported = dict(analysis.unsupported_ops())
        result.update(counted_gflops=total / 1e9,
                      unsupported_ops=unsupported,
                      uncalled_modules=sorted(analysis.uncalled_modules()),
                      status='partial' if unsupported else 'ok')
        if not unsupported:
            result['gflops'] = total / 1e9
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        for module, name, value in checkpoint_flags:
            setattr(module, name, value)
        for module, training in modes:
            module.training = training
    return result


def report_complexity(result):
    print(f"Params: {result['params']:,} ({result['params_m']:.4f} M); "
          f"trainable: {result['trainable_params']:,}")
    print(f"Complexity input: {result['input_shape']}; eval forward; "
          f"{result['flop_convention']}")
    if result['status'] == 'ok':
        print(f"GFLOPs: {result['gflops']:.4f}")
    elif result['status'] == 'partial':
        print(f"Counted GFLOPs (partial): {result['counted_gflops']:.4f}; "
              f"unsupported ops: {result['unsupported_ops']}")
    else:
        print(f"GFLOPs: {result['status']}. {result.get('error', '')}")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def main():
    from models.baselines import MODEL_REGISTRY, build_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelname', choices=tuple(MODEL_REGISTRY), default='Wave')
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--volume_size', type=int, nargs=3, default=(32, 128, 128))
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0, etc.')
    parser.add_argument('--output', default='model_complexity.json')
    args = parser.parse_args()
    model = build_model(args.modelname, args.base_channels, args.image_size,
                        args.volume_size).to(args.device)
    result = measure_complexity(model, input_shape(MODEL_REGISTRY[args.modelname],
                               args.image_size, args.volume_size))
    result.update(model=args.modelname, base_channels=args.base_channels)
    report_complexity(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Saved complexity: {output}')
    if result['status'] in ('failed', 'unavailable'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
