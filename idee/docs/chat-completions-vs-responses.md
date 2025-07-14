The main differences are:
- text block:
  * `"type": "input_text"` (responses)
  * `"type": "text"` (chat)
- image block:
  * `{"type": "input_image", "image_url": "..."}` or `{"type": "input_image", "file_id": "..."}` (responses)
  * `{"type": "image", "image_url": {"url": "..."}}` (chat), file is not supported)
- PDF file block:
  * `{"type": "input_file", "filename": "xxx.pdf", "file_data": "..."}` or `{"type": "input_file", "file_id": "..."}` (responses)
  * `{"type": "file", "file": {"filename": "xxx.pdf", "file_data": "..."}}` or `{"type": "file", "file": {"file_id": file.id}}` (chat)