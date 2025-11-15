import math
from typing import TYPE_CHECKING

import torch
import numpy as np
from torchvision.transforms import v2

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


class FrameEnhancers:
    """
    Manages frame enhancement (upscaling, colorization) models and processes.

    This class handles model loading, tiling for large images, and execution
    of various ONNX models (RealESRGAN, BSRGan, Deoldify, DDColor, etc.).
    """

    def __init__(self, models_processor: "ModelsProcessor"):
        """
        Initializes the FrameEnhancers class.

        Args:
            models_processor (ModelsProcessor): The central processor that manages
                                               model loading, unloading, and execution.
        """
        self.models_processor = models_processor
        self.current_enhancer_model = None  # Tracks the currently active enhancer model
        self.model_map = {
            # Maps user-facing names to internal model keys (used in models_processor)
            "RealEsrgan-x2-Plus": "RealEsrganx2Plus",
            "RealEsrgan-x4-Plus": "RealEsrganx4Plus",
            "BSRGan-x2": "BSRGANx2",
            "BSRGan-x4": "BSRGANx4",
            "UltraSharp-x4": "UltraSharpx4",
            "UltraMix-x4": "UltraMixx4",
            "RealEsr-General-x4v3": "RealEsrx4v3",
            "Deoldify-Artistic": "DeoldifyArt",
            "Deoldify-Stable": "DeoldifyStable",
            "Deoldify-Video": "DeoldifyVideo",
            "DDColor-Artistic": "DDColorArt",
            "DDColor": "DDcolor",
        }

    def unload_models(self):
        """
        Unloads the currently active enhancer model to free up VRAM.
        This is thread-safe, using the model_lock from models_processor.
        """
        with self.models_processor.model_lock:
            if self.current_enhancer_model:
                self.models_processor.unload_model(self.current_enhancer_model)
                self.current_enhancer_model = None

    def _run_model_with_lazy_build_check(
        self, model_name: str, ort_session, io_binding
    ):
        """
        Runs the ONNX session with IOBinding, handling TensorRT lazy build dialogs.

        This centralizes the try/finally logic for showing/hiding the build progress
        dialog and includes the critical synchronization step for CUDA or other devices.

        Args:
            model_name (str): The name of the model being run.
            ort_session: The ONNX Runtime session instance.
            io_binding: The pre-configured IOBinding object.
        """
        # Check if TensorRT is performing its one-time build for this model
        is_lazy_build = self.models_processor.check_and_clear_pending_build(model_name)
        if is_lazy_build:
            # Show a "please wait" dialog to the user
            self.models_processor.show_build_dialog.emit(
                "Finalizing TensorRT Build",
                f"Performing first-run inference for:\n{model_name}\n\nThis may take several minutes.",
            )

        try:
            # ⚠️ CRITICAL SYNCHRONIZATION POINT ⚠️
            # This ensures that the GPU (CUDA or other) has finished all previous
            # work before we run the model. This is vital in a multithreaded
            # environment to prevent race conditions or memory access errors.
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                # This handles synchronization for other execution providers (e.g., DirectML)
                # by synchronizing the custom sync vector.
                self.models_processor.syncvec.cpu()

            # Run the model using the pre-bound inputs and outputs
            ort_session.run_with_iobinding(io_binding)

        finally:
            # Always hide the dialog, even if the run fails
            if is_lazy_build:
                self.models_processor.hide_build_dialog.emit()

    def run_enhance_frame_tile_process(
        self, img, enhancer_type, tile_size=256, scale=1
    ):
        """
        Applies a selected enhancement model to an image using a tiling process.
        This is necessary for high-resolution images that don't fit into
        VRAM in one go.

        Args:
            img (torch.Tensor): The input image tensor (B, C, H, W).
            enhancer_type (str): The name of the enhancer to use (e.g., "RealEsrgan-x4-Plus").
            tile_size (int): The size of the square tiles to process.
            scale (int): The upscaling factor of the model (e.g., 2 for x2, 4 for x4).

        Returns:
            torch.Tensor: The enhanced (upscaled or colorized) image tensor.
        """
        # Model loading/unloading is now handled by control_actions.py (UI events).
        # We remove the redundant per-frame logic to prevent conflicts.
        # The 'current_enhancer_model' state is still set by control_actions.

        _, _, height, width = img.shape

        # --- 1. Calculate Tiling and Padding ---

        # Calculate the number of tiles needed
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # Calculate padding required to make the image dimensions divisible by tile_size
        pad_right = (tile_size - (width % tile_size)) % tile_size
        pad_bottom = (tile_size - (height % tile_size)) % tile_size

        # Apply padding to the image if necessary
        if pad_right != 0 or pad_bottom != 0:
            # Use 'constant' padding (black pixels)
            img = torch.nn.functional.pad(
                img, (0, pad_right, 0, pad_bottom), "constant", 0
            )

        # --- 2. Prepare Output Tensor and Select Model ---

        # Create an empty output tensor with the new scaled dimensions
        b, c, h, w = img.shape  # Get new padded dimensions
        output = torch.empty(
            (b, c, h * scale, w * scale),
            dtype=torch.float32,
            device=self.models_processor.device,
        ).contiguous()

        # Select the upscaling function based on the enhancer_type
        upscaler_functions = {
            "RealEsrgan-x2-Plus": self.run_realesrganx2,
            "RealEsrgan-x4-Plus": self.run_realesrganx4,
            "BSRGan-x2": self.run_bsrganx2,
            "BSRGan-x4": self.run_bsrganx4,
            "UltraSharp-x4": self.run_ultrasharpx4,
            "UltraMix-x4": self.run_ultramixx4,
            "RealEsr-General-x4v3": self.run_realesrx4v3,
        }

        fn_upscaler = upscaler_functions.get(enhancer_type)

        if not fn_upscaler:  # If the enhancer type is not a valid upscaler
            # Crop the original image back if padding was added, and return it
            if pad_right != 0 or pad_bottom != 0:
                img = v2.functional.crop(img, 0, 0, height, width)
            return img

        # --- 3. Process Tiles ---
        with torch.no_grad():  # Disable gradient calculation for inference
            # Process tiles
            for j in range(tiles_y):
                for i in range(tiles_x):
                    x_start, y_start = i * tile_size, j * tile_size
                    x_end, y_end = x_start + tile_size, y_start + tile_size

                    # Extract the input tile
                    input_tile = img[:, :, y_start:y_end, x_start:x_end].contiguous()

                    # Create an empty output tile with scaled dimensions
                    output_tile = torch.empty(
                        (
                            input_tile.shape[0],
                            input_tile.shape[1],
                            input_tile.shape[2] * scale,
                            input_tile.shape[3] * scale,
                        ),
                        dtype=torch.float32,
                        device=self.models_processor.device,
                    ).contiguous()

                    # Run the selected upscaler function on the tile
                    fn_upscaler(input_tile, output_tile)

                    # --- 4. Reassemble Output ---
                    # Calculate coordinates to place the output tile in the main output tensor
                    output_y_start, output_x_start = y_start * scale, x_start * scale
                    output_y_end, output_x_end = (
                        output_y_start + output_tile.shape[2],
                        output_x_start + output_tile.shape[3],
                    )
                    # Place the processed tile into the output tensor
                    output[
                        :, :, output_y_start:output_y_end, output_x_start:output_x_end
                    ] = output_tile

            # Crop the final output to remove the padding that was added
            if pad_right != 0 or pad_bottom != 0:
                output = v2.functional.crop(output, 0, 0, height * scale, width * scale)

        return output

    def _run_enhancer_model(
        self, model_name: str, image: torch.Tensor, output: torch.Tensor
    ):
        """
        Private helper to run any specified enhancer model.

        This function centralizes the logic for:
        1. Lazy-loading the model.
        2. Handling model loading errors with a robust fallback.
        3. Setting up IOBinding for inputs and outputs.
        4. Calling the synchronized execution function.

        Args:
            model_name (str): The internal key for the model (e.g., "RealEsrganx2Plus").
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        # Lazy-load the model if it's not already in memory
        if not self.models_processor.models[model_name]:
            self.models_processor.models[model_name] = self.models_processor.load_model(
                model_name
            )

        ort_session = self.models_processor.models[model_name]

        if not ort_session:
            # This fix ensures the output tensor is correctly populated instead
            # of containing uninitialized data (garbage).
            print(f"[WARN] Model {model_name} not loaded, skipping enhancer.")

            if image.shape == output.shape:
                # For 1:1 models (like colorizers), just copy the input
                output.copy_(image)
            else:
                # For upscalers, use bilinear interpolation as a fallback
                resized_image = torch.nn.functional.interpolate(
                    image, size=output.shape[-2:], mode="bilinear", align_corners=False
                )
                output.copy_(resized_image)
            return

        # Bind inputs and outputs directly to GPU memory pointers
        io_binding = ort_session.io_binding()

        # Bind input tensor
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=image.size(),
            buffer_ptr=image.data_ptr(),
        )
        # Bind output tensor
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=output.size(),
            buffer_ptr=output.data_ptr(),
        )

        # Run the model with lazy build handling and synchronization
        self._run_model_with_lazy_build_check(model_name, ort_session, io_binding)

    def run_realesrganx2(self, image, output):
        """
        Runs the RealEsrganx2Plus model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrganx2Plus", image, output)

    def run_realesrganx4(self, image, output):
        """
        Runs the RealEsrganx4Plus model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrganx4Plus", image, output)

    def run_realesrx4v3(self, image, output):
        """
        Runs the RealEsrx4v3 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("RealEsrx4v3", image, output)

    def run_bsrganx2(self, image, output):
        """
        Runs the BSRGANx2 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("BSRGANx2", image, output)

    def run_bsrganx4(self, image, output):
        """
        Runs the BSRGANx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("BSRGANx4", image, output)

    def run_ultrasharpx4(self, image, output):
        """
        Runs the UltraSharpx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("UltraSharpx4", image, output)

    def run_ultramixx4(self, image, output):
        """
        Runs the UltraMixx4 model on a given image tensor.
        This function is typically called per-tile by run_enhance_frame_tile_process.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("UltraMixx4", image, output)

    def run_deoldify_artistic(self, image, output):
        """
        Runs the DeoldifyArt (artistic colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyArt", image, output)

    def run_deoldify_stable(self, image, output):
        """
        Runs the DeoldifyStable (stable colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyStable", image, output)

    def run_deoldify_video(self, image, output):
        """
        Runs the DeoldifyVideo (video colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DeoldifyVideo", image, output)

    def run_ddcolor_artistic(self, image, output):
        """
        Runs the DDColorArt (artistic colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DDColorArt", image, output)

    def run_ddcolor(self, image, output):
        """
        Runs the DDcolor (general colorization) model on a given image tensor.

        Args:
            image (torch.Tensor): The input image (or tile) tensor.
            output (torch.Tensor): The pre-allocated output tensor to be filled.
        """
        self._run_enhancer_model("DDcolor", image, output)
