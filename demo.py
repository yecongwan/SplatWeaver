from pathlib import Path
import os
import sys
import torch

# -----------------------------------------------------------------------------
# Path setup (ensure project root is visible)
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video
from src.model.ply_export import export_ply
from src.model.model.splatweaver import SplatWeaver
from src.utils.image import process_image, visualize_expert_selection


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_ID = "Jeasco/SplatWeaver"
IMAGE_FOLDER = "examples/garden"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def load_images(image_folder: str, device: torch.device):
    """Load and preprocess images into model input tensor."""
    image_folder = Path(image_folder)

    image_paths = sorted([
        str(p) for p in image_folder.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    images = [process_image(p) for p in image_paths]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, H, W]

    return images, image_folder


def run_inference(model, images: torch.Tensor):
    """Run SplatWeaver inference."""
    with torch.no_grad():
        return model.inference((images + 1) * 0.5)


def save_results(gaussians, pred_context_pose, expert_selection, image_folder: Path, model, images):
    """Save all outputs (video, visualization, ply)."""

    b, v, _, h, w = images.shape  # kept intentionally external dependency behavior

    # Camera trajectories & rendering
    save_interpolated_video(
        pred_context_pose["extrinsic"],
        pred_context_pose["intrinsic"],
        b, h, w,
        gaussians,
        image_folder,
        model.decoder
    )

    # Expert selection visualization
    visualize_expert_selection(
        expert_selection[1],
        save_path=image_folder / "expert_selection.png"
    )

    # Export point cloud (Gaussian splats)
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        image_folder / "gaussians.ply"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Load model
    model = SplatWeaver.from_pretrained(MODEL_ID)
    model = model.to(DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # Load data
    images, image_folder = load_images(IMAGE_FOLDER, DEVICE)

    # Inference
    gaussians, pred_context_pose, expert_selection = run_inference(model, images)

    # Save outputs
    save_results(gaussians, pred_context_pose, expert_selection, image_folder, model, images)


if __name__ == "__main__":
    main()
