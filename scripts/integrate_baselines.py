from pathlib import Path
p=Path('main.py'); s=p.read_text(encoding='utf-8')
s=s.replace('import os\n','import os\nimport json\nfrom pathlib import Path\nfrom models.baselines import (MODEL_REGISTRY, build_model, validate_model_config,\n                              prediction_logits, segmentation_loss)\nfrom comparison_data import build_dataset\n',1)
s=s.replace('default="Wave", help=\'model name\'','default="Wave", choices=tuple(MODEL_REGISTRY), help=\'model name\'')
s=s.replace("    parser.add_argument('--base_channels'", "    parser.add_argument('--volume_size', type=int, nargs=3, default=(32,128,128), metavar=('D','H','W'))\n    parser.add_argument('--base_channels'",1)
s=s.replace('    arg = parser.parse_args()','    arg = parser.parse_args()\n    try:\n        validate_model_config(arg.modelname,arg.base_channels,arg.image_size,arg.volume_size)\n    except ValueError as exc:\n        parser.error(str(exc))',1)
a=s.index('        # 模型初始化'); b=s.index('        # Volume normalization',a)
s=s[:a]+'''        model = build_model(arg.modelname, arg.base_channels, arg.image_size, arg.volume_size)
        config = dict(vars(arg))
        config['input_dim'] = MODEL_REGISTRY[arg.modelname]
        config['initialization'] = 'checkpoint' if arg.pretrained else 'scratch (native constructor)'
        config['upstream'] = {p.parent.name: json.loads(p.read_text()) for p in Path('third_party').glob('*/UPSTREAM.json')}
        with open(os.path.join(save_dir, f'config_fold_{fold}.json'), 'w') as handle:
            json.dump(config, handle, indent=2)

'''+s[b:]
s=s.replace('dataset.ACDCDataset(arg.path_image, listTrainPatients,', 'build_dataset(arg.modelname, arg.path_image, listTrainPatients, volume_size=arg.volume_size,')
s=s.replace('dataset.ACDCDataset(arg.path_image, listValPatients,', 'build_dataset(arg.modelname, arg.path_image, listValPatients, volume_size=arg.volume_size,')
s=s.replace('            init_weights(model)  # Exactly once, rather than nested apply calls.',"            if arg.modelname in ('Wave','WaveDe'):\n                init_weights(model)  # Preserve native initialization of official baselines.")
s=s.replace('loss = criterion(output, target)','loss = segmentation_loss(criterion, output, target)')
s=s.replace('dice = dice_metric(output.detach(), target)','dice = dice_metric(prediction_logits(output, arg.modelname).detach(), target)')
s=s.replace('volume_dice.update(output.argmax(1), target, sample[\'volume_id\'])',"volume_dice.update(prediction_logits(output, arg.modelname).argmax(1), target, sample['volume_id'])")
s=s.replace('Train slice Dice:', 'Train sample Dice:')
p.write_text(s,encoding='utf-8')
p=Path('metrics.py'); s=p.read_text(encoding='utf-8'); s=s.replace("if logits.ndim != 4 or logits.shape[1] != 4:","if logits.ndim not in (4,5) or logits.shape[1] != 4:")
s=s.replace("if labels.ndim == 4 and labels.shape[1] == 1:","if labels.ndim == logits.ndim and labels.shape[1] == 1:")
p.write_text(s,encoding='utf-8')
p=Path('test.py'); s=p.read_text(encoding='utf-8')
s=s.replace('import dataset\n','import dataset\nfrom comparison_data import build_dataset\n')
s=s.replace("help='Must match the architecture used for training: Wave or WaveDe'","help='Must match the architecture used for training'")
s=s.replace("    parser.add_argument('--base_channels'", "    parser.add_argument('--volume_size', type=int, nargs=3, default=(32,128,128))\n    parser.add_argument('--base_channels'",1)
s=s.replace('validate_model_config(args.modelname, args.base_channels)','validate_model_config(args.modelname, args.base_channels, args.image_size, args.volume_size)')
s=s.replace('data = dataset.ACDCDataset(args.path_image, patients, mode=\'val\',',"data = build_dataset(args.modelname, args.path_image, patients, mode='val', volume_size=args.volume_size,")
s=s.replace('model = build_model(args.modelname, args.base_channels)','model = build_model(args.modelname, args.base_channels, args.image_size, args.volume_size)')
needle="            for i, volume_id in enumerate(sample['volume_id']):\n"
s=s.replace(needle,needle+'''                if MODEL_REGISTRY[args.modelname] == '3d':
                    shape = tuple(int(n) for n in data.volumes[volume_id]['volume_shape'])
                    restored = F.interpolate(logits[i:i+1], size=shape,
                                             mode='trilinear', align_corners=False)
                    prediction = restored.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    finish(volume_id, prediction, set(range(shape[0])))
                    continue
''')
p.write_text(s,encoding='utf-8')
