from .dgcnn import DGCNN


def build_encoder(arch, feat_dim, global_feat=True, **kwargs):
    if arch == 'dgcnn':
        model = DGCNN(feat_dim, global_feat=global_feat)
    else:
        raise NotImplementedError(f'{arch} is not supported')
    return model
