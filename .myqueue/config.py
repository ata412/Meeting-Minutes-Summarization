config = {
    'scheduler': 'slurm',
    'nodes': [
        # GPU node — H100 40GB
        ('gpu', {
            'cores': 8,
            'extra_args': [
                '--gpus-per-node=1',
                '--mem=80G',
            ],
        }),
        # CPU node — download / preprocess
        ('cpu', {
            'cores': 16,
            'extra_args': [
                '--mem=32G',
            ],
        }),
    ],
    'serial_python': (
        '/lustrefs/disk/project/zz991000-zdeva/zz991021/venv/bin/python3'
    ),
    'extra_args': [
        '--account=zz991021',
    ],
}
