Copyright (c) 2023 ETH Zurich, Ruben Mascaro. All rights reserved.

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Users should be careful about adopting these features in any commercial matters.

- SegFormer and MixTransformer: Copyright (c) 2021, NVIDIA Corporation,
  licensed under the NVIDIA Source Code License ([resources/license_segformer](resources/license_segformer)).
    - [mmseg_uda/models/decode_heads/segformer_head.py](mmseg_uda/models/decode_heads/segformer_head.py)
    - [mmseg_uda/models/backbones/mix_transformer.py](mmseg_uda/models/backbones/mix_transformer.py)
- DACS: Copyright (c) 2020, vikolss,
  licensed under the MIT License ([resources/license_dacs](resources/license_dacs)).
    - [mmseg_uda/utils/dacs_transforms.py](mmseg_uda/utils/dacs_transforms.py)
    - Parts of [mmseg_uda/models/uda/dacs.py](mmseg_uda/models/uda/dacs.py)
- DAFormer: Copyright (c) 2021-2022, ETH Zurich, Lukas Hoyer, licensed under the Apache License, Version 2.0 ([resources/license_daformer](resources/license_daformer)).
    - [mmseg_uda/models/decode_heads/daformer_head.py](mmseg_uda/models/decode_heads/daformer_head.py)
    - Parts of [mmseg_uda/datasets/uda_dataset.py](mmseg_uda/datasets/uda_dataset.py)
    - Parts of [mmseg_uda/models/uda/dacs.py](mmseg_uda/models/uda/dacs.py)
    - The base config files in configs/\_base\_/
    - The data preprocessing scripts in tools/convert_datasets/

The codebase heavily relies on:
- MMSegmentation: Copyright (c) 2020, The MMSegmentation Authors, licensed under the Apache License, Version 2.0 ([resources/license_mmseg](resources/license_mmseg))
- MIM: Copyright (c) 2021 OpenMMLab, licensed under the Apache License, Version 2.0 ([resources/license_mim](resources/license_mim))

