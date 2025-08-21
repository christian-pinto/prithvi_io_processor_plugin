# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ImagePrompt(BaseModel):

    data_format: Literal["b64_json", "bytes", "url"]
    """
    This is the data type for the input image
    """

    image_format: str
    """
    This is the image format (e.g., jpeg, png, etc.)
    """

    out_data_format: Literal["b64_json", "url"]

    data: Any
    """
    Input image data
    """


MultiModalPromptType = Union[ImagePrompt]

class ImageRequestOutput(BaseModel):
    """
    The output data of an image request to vLLM. 

    Args:
        type (str): The data content type [path, object]
        format (str): The image format (e.g., jpeg, png, etc.)
        data (Any): The resulting data.
    """

    type: Literal["path", "b64_json"]
    format: str
    data: str
    request_id: Optional[str] = None