import contextlib
import io
import os
import queue
import threading
import traceback

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Imported, NOT launched with subprocess. Inside a PyInstaller app sys.executable is
# the app itself (not Python), so "python reel_pipeline.py ..." just reopens the GUI.
import reel_pipeline

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class QueueWriter(io.TextIOBase):
    """Collects the pipeline's print() output so the GUI thread can show it."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)
        return len(s)

    def flush(self):
        pass


class ReelApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reel Pipeline GUI")
        self.geometry("640x640")

        self.input_path = ""
        self.output_path = ""
        self.msgs = queue.Queue()

        self.lbl_title = ctk.CTkLabel(self, text="Reel Auto-Editor", font=("Arial", 22, "bold"))
        self.lbl_title.pack(pady=(18, 10))

        self.btn_input = ctk.CTkButton(self, text="Select Input Video", command=self.select_input)
        self.btn_input.pack(pady=8)

        self.lbl_input = ctk.CTkLabel(self, text="No video selected", text_color="gray")
        self.lbl_input.pack(pady=2)

        self.template_var = ctk.StringVar(value="Reel_FastPaced")
        self.menu_template = ctk.CTkOptionMenu(
            self, values=list(reel_pipeline.PRESETS), variable=self.template_var)
        self.menu_template.pack(pady=8)

        self.transcribe_var = ctk.BooleanVar(value=False)
        self.chk_transcribe = ctk.CTkCheckBox(
            self, text="Enable Subtitles (--transcribe)", variable=self.transcribe_var)
        self.chk_transcribe.pack(pady=8)

        self.btn_process = ctk.CTkButton(
            self, text="Start Processing", fg_color="green", command=self.start_thread)
        self.btn_process.pack(pady=10)

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate", width=360)
        self.progress.pack(pady=4)
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(self, text="Status: Ready", font=("Arial", 14))
        self.lbl_status.pack(pady=6)

        self.log = ctk.CTkTextbox(self, width=590, height=250)
        self.log.pack(padx=20, pady=(4, 16), fill="both", expand=True)

        self.after(150, self.poll)

    # ------------------------------------------------------------------ UI
    def select_input(self):
        file_selected = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv")])
        if file_selected:
            self.input_path = file_selected
            self.lbl_input.configure(text=os.path.basename(file_selected), text_color="white")
            self.output_path = os.path.splitext(file_selected)[0] + "_edited.mp4"

    def append_log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def start_thread(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Pehle video file select karein!")
            return

        args = [self.input_path, "-o", self.output_path, "--template", self.template_var.get()]
        if os.path.exists(self.output_path):
            if not messagebox.askyesno(
                    "File exists", f"{os.path.basename(self.output_path)} already exists.\nOverwrite it?"):
                return
            args.append("--force")
        if self.transcribe_var.get():
            args.append("--transcribe")

        self.log.delete("1.0", "end")
        self.btn_process.configure(state="disabled")
        self.lbl_status.configure(text="Processing... Please wait.")
        self.progress.start()
        threading.Thread(target=self.run_pipeline, args=(args,), daemon=True).start()

    # ------------------------------------------------------- worker thread
    def run_pipeline(self, args):
        """Runs off the UI thread. Never touches widgets: it only puts messages on a queue."""
        rc, err = 1, None
        writer = QueueWriter(self.msgs)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                rc = reel_pipeline.main(args)
        except reel_pipeline.PipelineError as e:
            err = str(e)
        except SystemExit as e:  # argparse errors
            err = f"Invalid arguments (exit {e.code}). See log."
        except Exception:
            err = traceback.format_exc()
        self.msgs.put(("done", rc, err))

    # ------------------------------------------------------------ UI thread
    def poll(self):
        try:
            while True:
                item = self.msgs.get_nowait()
                if isinstance(item, tuple):
                    self.finish(item[1], item[2])
                else:
                    self.append_log(item)
        except queue.Empty:
            pass
        self.after(150, self.poll)

    def finish(self, rc, err):
        self.progress.stop()
        self.progress.set(0)
        self.btn_process.configure(state="normal")
        if err:
            self.append_log(f"\nERROR: {err}\n")
            self.lbl_status.configure(text="Status: Error Occurred!")
            messagebox.showerror("Error", err[-700:])
        elif rc != 0:
            self.lbl_status.configure(text="Status: Finished with warnings")
            messagebox.showwarning(
                "Check output", f"Video bana, par validation mein problem mili. Log dekhein.\n{self.output_path}")
        else:
            self.lbl_status.configure(text="Status: Completed Successfully!")
            messagebox.showinfo("Success", f"Video edit ho gaya!\nSaved at: {self.output_path}")


if __name__ == "__main__":
    app = ReelApp()
    app.mainloop()
