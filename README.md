# Adnexify
AI Powered Dynamic Content Creation.

## Setup Instructions

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv new_env_name
    source new_env_name/bin/activate
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configuration:**

    - In `object_overlay.py`, update the path of `"lama_ckpt"` to your system path after downloading required checkpoint.
    - Download checkpoint 'sam_vit_b_01ec64.pth' and put in folder `sam_checkpoints`
    - In `ad_creation.py`, provide your `OPENAI_API_KEY`.

4. **Run the Application:**

    To start the app, use the following command:

    ```bash
    python app.py
    ```
