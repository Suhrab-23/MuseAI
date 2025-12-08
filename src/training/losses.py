"""
Loss functions for style transfer training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """
    Content loss using VGG features.
    Measures L2 distance in feature space.
    """

    def __init__(self, layer='relu4_1'):
        super(ContentLoss, self).__init__()
        self.layer = layer

    def forward(self, content_features, stylized_features):
        """
        Compute content loss.

        Args:
            content_features: Features from content image (dict of tensors)
            stylized_features: Features from stylized image (dict of tensors)
        """
        content_feat = content_features[self.layer]
        stylized_feat = stylized_features[self.layer]

        loss = F.mse_loss(stylized_feat, content_feat)
        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrices.
    Measures style similarity across multiple layers.
    """

    def __init__(self, layers=None):
        super(StyleLoss, self).__init__()
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        self.layers = layers

    def gram_matrix(self, features):
        """
        Compute Gram matrix.

        Args:
            features: Feature tensor (B, C, H, W)

        Returns:
            Gram matrix (B, C, C)
        """
        B, C, H, W = features.size()

        # (B, C, H*W)
        features_reshaped = features.view(B, C, H * W)

        # (B, C, C)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))

        # Normalize only by spatial size so magnitudes are not too tiny
        gram = gram / (H * W)

        return gram

    def forward(self, style_features, stylized_features):
        """
        Compute style loss across multiple layers.

        Args:
            style_features: Features from style image (dict)
            stylized_features: Features from stylized image (dict)

        Returns:
            Style loss (scalar)
        """
        loss = 0.0

        for layer in self.layers:
            style_feat = style_features[layer]
            stylized_feat = stylized_features[layer]

            style_gram = self.gram_matrix(style_feat)
            stylized_gram = self.gram_matrix(stylized_feat)

            layer_loss = F.mse_loss(stylized_gram, style_gram)
            loss += layer_loss

        # NOTE: we do NOT divide by len(self.layers) here, so style is strong
        return loss


class IdentityLoss(nn.Module):
    """
    Identity preservation loss using FaceNet embeddings.
    Ensures the stylized face maintains the subject's identity.
    """

    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, content_embedding, stylized_embedding):
        """
        Compute identity loss using cosine distance.

        Args:
            content_embedding: Embedding from content image (B, D)
            stylized_embedding: Embedding from stylized image (B, D)
        """
        similarity = (content_embedding * stylized_embedding).sum(dim=1).mean()
        loss = 1.0 - similarity
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total variation loss for image smoothness.
    Reduces noise and artifacts.
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        """
        Compute TV loss.

        Args:
            image: Image tensor (B, C, H, W)
        """
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])

        tv_loss = diff_h.mean() + diff_w.mean()
        return tv_loss


class CombinedLoss(nn.Module):
    """
    Combined loss for style transfer training.
    Supports per-artist loss weights so Rembrandt and Picasso
    can behave differently without changing the architecture.
    """

    def __init__(self, config):
        super(CombinedLoss, self).__init__()

        self.content_loss = ContentLoss(layer=config['model']['content_layer'])
        self.style_loss = StyleLoss(layers=config['model']['style_layers'])
        self.identity_loss = IdentityLoss()
        self.tv_loss = TotalVariationLoss()

        lw_cfg = config['training']['loss_weights']
        # If user didn't nest under `default`, fall back to single dict
        self.default_weights = lw_cfg.get('default', lw_cfg)
        self.per_artist_weights = lw_cfg.get('per_artist', {})

    def _get_weights_for_artist(self, artist_name: str):
        """
        Get loss weights for a given artist.
        Falls back to default for any missing keys.
        """
        base = self.default_weights
        overrides = self.per_artist_weights.get(artist_name, {})

        return {
            'content': overrides.get('content', base['content']),
            'style': overrides.get('style', base['style']),
            'identity': overrides.get('identity', base['identity']),
            'tv': overrides.get('tv', base['tv']),
        }

    def forward(self, features, stylized_img, artist_name: str):
        """
        Compute total loss.

        Args:
            features: dict with:
                'content_features', 'style_features',
                'stylized_features',
                'content_embedding', 'stylized_embedding'
            stylized_img: stylized image for TV loss (B, 3, H, W)
            artist_name: string name, e.g. "picasso" or "rembrandt"
        """
        w = self._get_weights_for_artist(artist_name)

        content_loss = self.content_loss(
            features['content_features'],
            features['stylized_features']
        )

        style_loss = self.style_loss(
            features['style_features'],
            features['stylized_features']
        )

        identity_loss = self.identity_loss(
            features['content_embedding'],
            features['stylized_embedding']
        )

        tv_loss = self.tv_loss(stylized_img)

        total_loss = (
            w['content'] * content_loss +
            w['style'] * style_loss +
            w['identity'] * identity_loss +
            w['tv'] * tv_loss
        )

        return {
            'total': total_loss,
            'content': content_loss.item(),
            'style': style_loss.item(),
            'identity': identity_loss.item(),
            'tv': tv_loss.item()
        }


def test_losses():
    """Quick sanity check."""
    import yaml

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, C, H, W = 2, 512, 64, 64

    features = {
        'content_features': {
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'style_features': {
            'relu1_1': torch.randn(B, 64, 256, 256).to(device),
            'relu2_1': torch.randn(B, 128, 128, 128).to(device),
            'relu3_1': torch.randn(B, 256, 64, 64).to(device),
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'stylized_features': {
            'relu1_1': torch.randn(B, 64, 256, 256).to(device),
            'relu2_1': torch.randn(B, 128, 128, 128).to(device),
            'relu3_1': torch.randn(B, 256, 64, 64).to(device),
            'relu4_1': torch.randn(B, C, H, W).to(device)
        },
        'content_embedding': torch.randn(B, 512).to(device),
        'stylized_embedding': torch.randn(B, 512).to(device)
    }

    stylized_img = torch.rand(B, 3, 512, 512).to(device)

    criterion = CombinedLoss(config).to(device)
    losses = criterion(features, stylized_img, artist_name="rembrandt")

    print("Loss Test:")
    for name, value in losses.items():
        if name == 'total':
            print(f"  {name}: {value.item():.4f}")
        else:
            print(f"  {name}: {value:.4f}")


if __name__ == '__main__':
    test_losses()
