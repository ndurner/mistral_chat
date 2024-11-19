import gradio as gr
import base64
import os 
from mistralai import Mistral
import json
import fitz
from PIL import Image
import io
from settings_mgr import generate_download_settings_js, generate_upload_settings_js

from doc2json import process_docx

dump_controls = False
log_to_console = False

temp_files = []

def encode_image(image_data):
    """Generates a prefix for image base64 data in the required format for the
    four known image formats: png, jpeg, gif, and webp.

    Args:
    image_data: The image data, encoded in base64.

    Returns:
    A string containing the prefix.
    """

    # Get the first few bytes of the image data.
    magic_number = image_data[:4]
  
    # Check the magic number to determine the image type.
    if magic_number.startswith(b'\x89PNG'):
        image_type = 'png'
    elif magic_number.startswith(b'\xFF\xD8'):
        image_type = 'jpeg'
    elif magic_number.startswith(b'GIF89a'):
        image_type = 'gif'
    elif magic_number.startswith(b'RIFF'):
        if image_data[8:12] == b'WEBP':
            image_type = 'webp'
        else:
            # Unknown image type.
            raise Exception("Unknown image type")
    else:
        # Unknown image type.
        raise Exception("Unknown image type")

    return f"data:image/{image_type};base64,{base64.b64encode(image_data).decode('utf-8')}"

def process_pdf_img(pdf_fn: str):
    pdf = fitz.open(pdf_fn)
    message_parts = []

    for page in pdf.pages():
        # Create a transformation matrix for rendering at the calculated scale
        mat = fitz.Matrix(0.6, 0.6)
        
        # Render the page to a pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode image to base64
        base64_encoded = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Construct the data URL
        image_url = f"data:image/png;base64,{base64_encoded}"
        
        # Append the message part
        message_parts.append({
            "type": "text",
            "text": f"Page {page.number} of file '{pdf_fn}'"
        })
        message_parts.append({
            "type": "image_url",
            "image_url": image_url
        })

    pdf.close()

    return message_parts

def encode_file(fn: str) -> list:
    user_msg_parts = []

    if fn.endswith(".docx"):
        user_msg_parts.append({"type": "text", "text": process_docx(fn)})
    elif fn.endswith(".pdf"):
        user_msg_parts.extend(process_pdf_img(fn))
    else:
        with open(fn, mode="rb") as f:
            content = f.read()

        isImage = False
        if isinstance(content, bytes):
            try:
                # try to add as image
                content = encode_image(content)
                isImage = True
            except:
                # not an image, try text
                content = content.decode('utf-8', 'replace')
        else:
            content = str(content)

        if isImage:
            user_msg_parts.append({"type": "image_url", "image_url": content})
        else:
            user_msg_parts.append({"type": "text", "text": content})

    return user_msg_parts

def bot(message, history, mistral_key, system_prompt, seed, temperature, max_tokens, model):
    try:
        client = Mistral(
            api_key=mistral_key
        )

        history_mistral_format = []
        user_msg_parts = []

        if system_prompt:
            history_mistral_format.append({"role": "system", "content": system_prompt})

        for human, assi in history:
            if human is not None:
                if type(human) is tuple:
                    user_msg_parts.extend(encode_file(human[0]))
                else:
                    user_msg_parts.append({"type": "text", "text": human})

            if assi is not None:
                if user_msg_parts:
                    history_mistral_format.append({"role": "user", "content": user_msg_parts})
                    user_msg_parts = []

                history_mistral_format.append({"role": "assistant", "content": assi})

        if message["text"]:
            user_msg_parts.append({"type": "text", "text": message["text"]})
        if message["files"]:
            for file in message["files"]:
                user_msg_parts.extend(encode_file(file))
        history_mistral_format.append({"role": "user", "content": user_msg_parts})

        if log_to_console:
            print(f"br_prompt: {str(history_mistral_format)}")

        response = client.chat.stream(
            model=model,
            messages=history_mistral_format,
            temperature=temperature,
            max_tokens=max_tokens
        )

        partial_response = ""
        for chunk in response:
            if chunk.data.choices:
                txt = chunk.data.choices[0].delta.content
                if txt:
                    partial_response += txt
                    yield partial_response

        if log_to_console:
            print(f"br_result: {str(history)}")

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")

def undo(history):
    history.pop()
    return history

def dump(history):
    return str(history)

def load_settings():  
    # Dummy Python function, actual loading is done in JS  
    pass  

def save_settings(acc, sec, prompt, temp, tokens, model):  
    # Dummy Python function, actual saving is done in JS  
    pass  

def import_history(history, file):
    with open(file.name, mode="rb") as f:
        content = f.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', 'replace')
        else:
            content = str(content)
    os.remove(file.name)

    # Deserialize the JSON content
    import_data = json.loads(content)

    # Check if 'history' key exists for backward compatibility
    if 'history' in import_data:
        history = import_data['history']
        system_prompt.value = import_data.get('system_prompt', '')  # Set default if not present
    else:
        # Assume it's an old format with only history data
        history = import_data

    return history, system_prompt.value

with gr.Blocks(delete_cache=(86400, 86400)) as demo:
    gr.Markdown("# Mistral Chat")
    with gr.Accordion("Startup"):
        gr.Markdown("""Use of this interface permitted under the terms and conditions of the 
                    [MIT license](https://github.com/ndurner/mistral_chat/blob/main/LICENSE).
                    Third party terms and conditions apply. This app and the AI models may make mistakes, so verify any outputs.""")

        mistral_key = gr.Textbox(label="Mistral API Key", elem_id="mistral_key")
        model = gr.Dropdown(label="Model", value="pixtral-large-latest", allow_custom_value=True, elem_id="model",
                            choices=["pixtral-large-latest", "mistral-large-latest", "pixtral-12b-2409"])
        system_prompt = gr.TextArea("You are a helpful yet diligent AI assistant. Answer faithfully and factually correct. Respond with 'I do not know' if uncertain.", 
                                  label="System Prompt", lines=3, max_lines=250, elem_id="system_prompt")  
        seed = gr.Textbox(label="Seed", elem_id="seed")
        temp = gr.Slider(0, 1, label="Temperature", elem_id="temp", value=0.7)
        max_tokens = gr.Slider(1, 4096, label="Max. Tokens", elem_id="max_tokens", value=800)
        save_button = gr.Button("Save Settings")  
        load_button = gr.Button("Load Settings")  
        dl_settings_button = gr.Button("Download Settings")
        ul_settings_button = gr.Button("Upload Settings")

        load_button.click(load_settings, js="""  
            () => {  
                let elems = ['#mistral_key textarea', '#system_prompt textarea', '#seed textarea', '#temp input', '#max_tokens input', '#model'];
                elems.forEach(elem => {
                    let item = document.querySelector(elem);
                    let event = new InputEvent('input', { bubbles: true });
                    item.value = localStorage.getItem(elem.split(" ")[0].slice(1)) || '';
                    item.dispatchEvent(event);
                });
            }  
        """)

        save_button.click(save_settings, [mistral_key, system_prompt, seed, temp, max_tokens, model], js="""  
            (key, sys, seed, temp, ntok, model) => {  
                localStorage.setItem('mistral_key', key);  
                localStorage.setItem('system_prompt', sys);  
                localStorage.setItem('seed', seed);  
                localStorage.setItem('temp', document.querySelector('#temp input').value);  
                localStorage.setItem('max_tokens', document.querySelector('#max_tokens input').value);  
                localStorage.setItem('model', model);  
            }  
        """) 

        control_ids = [('mistral_key', '#mistral_key textarea'),
                      ('system_prompt', '#system_prompt textarea'),
                      ('seed', '#seed textarea'),
                      ('temp', '#temp input'),
                      ('max_tokens', '#max_tokens input'),
                      ('model', '#model')]
        controls = [mistral_key, system_prompt, seed, temp, max_tokens, model]

        dl_settings_button.click(None, controls, js=generate_download_settings_js("mistral_chat_settings.bin", control_ids))
        ul_settings_button.click(None, None, None, js=generate_upload_settings_js(control_ids))

    chat = gr.ChatInterface(fn=bot, multimodal=True, additional_inputs=controls, autofocus=False)
    chat.textbox.file_count = "multiple"
    chatbot = chat.chatbot
    chatbot.show_copy_button = True
    chatbot.height = 450

    if dump_controls:
        with gr.Row():
            dmp_btn = gr.Button("Dump")
            txt_dmp = gr.Textbox("Dump")
            dmp_btn.click(dump, inputs=[chatbot], outputs=[txt_dmp])

    with gr.Accordion("Import/Export", open=False):
        import_button = gr.UploadButton("History Import")
        export_button = gr.Button("History Export")
        export_button.click(lambda: None, [chatbot, system_prompt], js="""
            (chat_history, system_prompt) => {
                const export_data = {
                    history: chat_history,
                    system_prompt: system_prompt
                };
                const history_json = JSON.stringify(export_data);
                const blob = new Blob([history_json], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'chat_history.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
            """)
        dl_button = gr.Button("File download")
        dl_button.click(lambda: None, [chatbot], js="""
            (chat_history) => {
                const languageToExt = {
                    'python': 'py',
                    'javascript': 'js',
                    'typescript': 'ts',
                    'csharp': 'cs',
                    'ruby': 'rb',
                    'shell': 'sh',
                    'bash': 'sh',
                    'markdown': 'md',
                    'yaml': 'yml',
                    'rust': 'rs',
                    'golang': 'go',
                    'kotlin': 'kt'
                };

                const contentRegex = /```(?:([^\\n]+)?\\n)?([\\s\\S]*?)```/;
                const match = contentRegex.exec(chat_history[chat_history.length - 1][1]);
                
                if (match && match[2]) {
                    const specifier = match[1] ? match[1].trim() : '';
                    const content = match[2];
                    
                    let filename = 'download';
                    let fileExtension = 'txt'; // default

                    if (specifier) {
                        if (specifier.includes('.')) {
                            // If specifier contains a dot, treat it as a filename
                            const parts = specifier.split('.');
                            filename = parts[0];
                            fileExtension = parts[1];
                        } else {
                            // Use mapping if exists, otherwise use specifier itself
                            const langLower = specifier.toLowerCase();
                            fileExtension = languageToExt[langLower] || langLower;
                            filename = 'code';
                        }
                    }

                    const blob = new Blob([content], {type: 'text/plain'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${filename}.${fileExtension}`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }
            }
        """)
        import_button.upload(import_history, inputs=[chatbot, import_button], outputs=[chatbot, system_prompt])

demo.unload(lambda: [os.remove(file) for file in temp_files])
demo.launch()