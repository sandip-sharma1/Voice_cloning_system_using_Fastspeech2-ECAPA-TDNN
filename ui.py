# Full graphical user interface for recording, embedding extraction, and synthesis
# Fixed: QMediaContent wrapping for setMedia() - no more TypeError on macOS
# Added: Pause / Resume button that works for both processed samples and synthesized audio
"""
this user interface is built using PyQt5 and provides the following features:
1. Record new speaker using microphone and generate embedding
2. List available speakers and their embeddings, with option to play processed sample
3. Synthesize speech from text using selected speaker embedding, with pitch/energy/duration controls
4. Playback controls for synthesized audio (play/pause/stop)
imp note: Make sure to run this with the virtual environment activated that has all dependencies installed, and ensure that the generate.py script is in the same directory or adjust the path accordingly.
---most importantly it can only vary energy,pitch and duration of synthesized audio by 0.1 so if varying those parameters
on the scale lower than that the generate.py must be used...or tuning of ui.py must be done
"""

"""
most important note: this code is made to run on cpu because it was developed on Macbook Air m1 which has no cuda support. If you want to run it on gpu, 
you may need to adjust the full_pipeline function to ensure that the embedding extraction runs on the gpu. Specifically, you would need to move the audio 
tensor to the gpu before passing it to the model, and also ensure that the model itself is loaded onto the gpu. This would involve changing lines in the full_pipeline 
function where the audio is processed and where the model is loaded. Additionally, make sure that your environment has access to a compatible GPU and that PyTorch is installed with CUDA support.
"""



# second version  (or full_tts_gui_fixed_final_with_pause button )
# Fixed: Added missing update_rec_label method
#final version has all the features ......

import sys
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false;qt5ct.debug=false"

import torch
import torchaudio
import numpy as np
import threading
import time
import subprocess
import glob

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QListWidget,
    QMessageBox, QGroupBox, QStatusBar, QTextEdit, QComboBox,
    QSlider, QTabWidget, QFileDialog, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import QUrl, Qt, pyqtSignal, QObject
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QFont, QPixmap

import sounddevice as sd

from model.pre_trained.ecapa_tdnn_loader import get_ECAPA_TDNN_MODEL, speaker_embedding_extractor
from audio.tools import trim_silence

# ── Directory constants ──────────────────────────────────────────────────────
SEEN_DIR   = "embeddings/LibriTTS"
UNSEEN_DIR = "unseenembeddings"

# ── Embedding file helpers ───────────────────────────────────────────────────

def get_embedding_path(directory, sid):
    """
    Seen speakers  → {sid}.ecapa_averaged_embedding
    Unseen speakers → {sid}.pt
    """
    if directory == SEEN_DIR:
        p = os.path.join(directory, f"{sid}.ecapa_averaged_embedding")
        if os.path.exists(p):
            return p
        # Fallback to .pt if averaged embedding not found
        return os.path.join(directory, f"{sid}.pt")
    else:
        return os.path.join(directory, f"{sid}.pt")


def load_embedding(directory, sid):
    """Load the correct embedding tensor for the given speaker."""
    path = get_embedding_path(directory, sid)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding not found: {path}")
    return torch.load(path), path


# ── Pipeline ─────────────────────────────────────────────────────────────────

def full_pipeline(audio_path, speaker_id, output_dir=SEEN_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, f"{speaker_id}_processed.wav")
    out_pt  = os.path.join(output_dir, f"{speaker_id}.pt")

    audio, sr = torchaudio.load(audio_path)
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio / peak
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
        sr = 16000
    audio = trim_silence(audio)
    audio = audio.squeeze(0).to(device).float()

    torchaudio.save(out_wav, audio.cpu().unsqueeze(0), sample_rate=sr)

    model = get_ECAPA_TDNN_MODEL(device=device)
    model.eval()
    with torch.no_grad():
        embedding = speaker_embedding_extractor(model, audio.unsqueeze(0))

    torch.save(embedding.cpu(), out_pt)
    return out_wav, out_pt, embedding


def cosine_similarity_pct(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Cosine similarity mapped to 0–100 %."""
    e1 = torch.nn.functional.normalize(e1.view(1, -1).float(), p=2, dim=1)
    e2 = torch.nn.functional.normalize(e2.view(1, -1).float(), p=2, dim=1)
    cos = float((e1 * e2).sum())           # -1 … 1
    return round((cos + 1) / 2 * 100, 2)  # 0 … 100 %


def next_global_id():
    """Return next integer ID that is unique across both directories."""
    nums = []
    for d in [SEEN_DIR, UNSEEN_DIR]:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            stem = os.path.splitext(f)[0]
            if f.endswith(".pt") and not stem.endswith("_processed") and stem.isdigit():
                nums.append(int(stem))
    return str(max(nums) + 1) if nums else "1"


# ────────────────────────────────────────────────────────────────────────────

class RecorderSignals(QObject):
    update_time    = pyqtSignal(float)
    error_message  = pyqtSignal(str)
    mel_ready      = pyqtSignal(str)   # emits path to mel PNG


class TTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Cloning System")
        self.resize(1200, 860)

        os.makedirs("createdataset", exist_ok=True)
        os.makedirs(SEEN_DIR,   exist_ok=True)
        os.makedirs(UNSEEN_DIR, exist_ok=True)

        self.recording     = False
        self.audio_chunks  = []
        self.sample_rate   = 44100
        self.start_t       = 0
        self.record_thread = None
        self.selected_audio_file = None

        self.rec_signals = RecorderSignals()
        self.rec_signals.update_time.connect(self.update_rec_label)
        self.rec_signals.error_message.connect(self.show_rec_error)
        self.rec_signals.mel_ready.connect(self._display_mel)

        self.player = QMediaPlayer()
        self.last_synthesized_file = None

        self.init_ui()
        self.refresh_speakers()

    # ════════════════════════════════════════════════════════════════════════
    # UI
    # ════════════════════════════════════════════════════════════════════════

    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_lay = QVBoxLayout(cw)

        tabs = QTabWidget()
        main_lay.addWidget(tabs)

        self._build_record_tab(tabs)
        self._build_speakers_tab(tabs)
        self._build_synthesize_tab(tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — make sure venv is active")

    # ── Record tab ───────────────────────────────────────────────────────────
    def _build_record_tab(self, tabs):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        g = QGroupBox(" 1. Record new speaker or use existing audio ")
        f = QFormLayout()

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("example: sandip_new_01  (optional friendly name)")
        f.addRow("Friendly name:", self.filename_edit)

        note = QLabel(
            "New speakers are saved to  unseenembeddings/  "
            "with an ID continuing from the last seen speaker."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; font-style:italic;")
        f.addRow(note)

        browse_row = QHBoxLayout()
        self.btn_browse = QPushButton("📂 Browse existing audio file")
        self.btn_browse.clicked.connect(self.browse_audio_file)
        browse_row.addWidget(self.btn_browse)
        self.lbl_source = QLabel("No source selected")
        self.lbl_source.setStyleSheet("color:#666; font-style:italic;")
        browse_row.addWidget(self.lbl_source)
        browse_row.addStretch()
        f.addRow(browse_row)

        btnrow = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start Recording (mic)")
        self.btn_start.clicked.connect(self.start_rec)
        self.btn_stop = QPushButton("⏹ Stop & Generate Embedding")
        self.btn_stop.clicked.connect(self.stop_rec)
        self.btn_stop.setEnabled(False)
        btnrow.addWidget(self.btn_start)
        btnrow.addWidget(self.btn_stop)
        f.addRow(btnrow)

        self.lbl_rec = QLabel("Ready")
        self.lbl_rec.setAlignment(Qt.AlignCenter)
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")
        f.addRow(self.lbl_rec)

        g.setLayout(f)
        lay.addWidget(g)
        tabs.addTab(tab, "Record")

    # ── Speakers tab ─────────────────────────────────────────────────────────
    def _build_speakers_tab(self, tabs):
        tab = QWidget()
        lay = QVBoxLayout(tab)

        # seen
        g_seen = QGroupBox(f" Seen Speakers  [{SEEN_DIR}]  —  uses .pt ")
        v_seen = QVBoxLayout()
        self.lst_seen = QListWidget()
        self.lst_seen.itemDoubleClicked.connect(lambda i: self._on_dblclick(i, SEEN_DIR))
        self.lst_seen.itemSelectionChanged.connect(
            lambda: self._on_sel_changed(self.lst_seen, SEEN_DIR))
        v_seen.addWidget(self.lst_seen)
        h = QHBoxLayout()
        b = QPushButton("↻ Refresh"); b.clicked.connect(self.refresh_speakers); h.addWidget(b)
        b2 = QPushButton("▶ Play"); b2.clicked.connect(
            lambda: self._play_selected(self.lst_seen, SEEN_DIR)); h.addWidget(b2)
        v_seen.addLayout(h)
        g_seen.setLayout(v_seen)
        lay.addWidget(g_seen)

        # unseen
        g_unseen = QGroupBox(f" Unseen Speakers  [{UNSEEN_DIR}]  —  uses .pt ")
        v_unseen = QVBoxLayout()
        self.lst_unseen = QListWidget()
        self.lst_unseen.itemDoubleClicked.connect(lambda i: self._on_dblclick(i, UNSEEN_DIR))
        self.lst_unseen.itemSelectionChanged.connect(
            lambda: self._on_sel_changed(self.lst_unseen, UNSEEN_DIR))
        v_unseen.addWidget(self.lst_unseen)
        h2 = QHBoxLayout()
        b3 = QPushButton("▶ Play"); b3.clicked.connect(
            lambda: self._play_selected(self.lst_unseen, UNSEEN_DIR)); h2.addWidget(b3)
        b4 = QPushButton("🗑 Delete"); b4.clicked.connect(self._delete_unseen); h2.addWidget(b4)
        v_unseen.addLayout(h2)
        g_unseen.setLayout(v_unseen)
        lay.addWidget(g_unseen)

        # shared details / note
        self.btn_pause_listen = QPushButton("⏯ Pause / Resume Listening")
        self.btn_pause_listen.clicked.connect(self.toggle_pause_listen)
        self.btn_pause_listen.setEnabled(False)
        lay.addWidget(self.btn_pause_listen)

        self.txt_details = QTextEdit(readOnly=True)
        self.txt_details.setMaximumHeight(150)
        lay.addWidget(QLabel("Embedding Details:"))
        lay.addWidget(self.txt_details)

        lay.addWidget(QLabel("Speaker Note:"))
        self.txt_note = QTextEdit()
        self.txt_note.setMaximumHeight(70)
        self.txt_note.setPlaceholderText("Add a note for this speaker.")
        lay.addWidget(self.txt_note)

        btn_save = QPushButton("💾 Save Note")
        btn_save.clicked.connect(self.save_speaker_note)
        lay.addWidget(btn_save)

        tabs.addTab(tab, "Speakers")

    # ── Synthesize tab ───────────────────────────────────────────────────────
    def _build_synthesize_tab(self, tabs):
        tab = QWidget()
        tab_lay = QVBoxLayout(tab)

        g = QGroupBox(" 3. Text → Speech Synthesis ")
        outer = QVBoxLayout()

        # ── Speaker controls (full width) ────────────────────────────────
        f_top = QFormLayout()

        self.cmb_set = QComboBox()
        self.cmb_set.addItem("Multi-Speaker ", SEEN_DIR)
        self.cmb_set.addItem("Zero-Shot ", UNSEEN_DIR)
        self.cmb_set.currentIndexChanged.connect(self._on_set_changed)
        f_top.addRow("Mode:", self.cmb_set)

        self.cmb_speaker = QComboBox()
        f_top.addRow("Select speaker:", self.cmb_speaker)
        outer.addLayout(f_top)

        # ── Text input  +  Mel spectrogram side by side ──────────────────
        text_mel_row = QHBoxLayout()

        # Left: text input
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Text to speak:"))
        self.txt_input = QTextEdit()
        self.txt_input.setPlaceholderText("Type your sentence here...")
        self.txt_input.setMinimumHeight(120)
        left_col.addWidget(self.txt_input)
        text_mel_row.addLayout(left_col, stretch=1)

        # Right: mel spectrogram panel
        right_col = QVBoxLayout()
        mel_header = QHBoxLayout()
        mel_header.addWidget(QLabel("Mel Spectrogram:"))
        self.btn_clear_mel = QPushButton("✕ Clear")
        self.btn_clear_mel.setFixedWidth(70)
        self.btn_clear_mel.setStyleSheet("font-size:11px; padding:2px;")
        self.btn_clear_mel.clicked.connect(self._clear_mel)
        mel_header.addWidget(self.btn_clear_mel)
        mel_header.addStretch()
        right_col.addLayout(mel_header)

        self.lbl_mel = QLabel()
        self.lbl_mel.setAlignment(Qt.AlignCenter)
        self.lbl_mel.setMinimumSize(320, 120)
        self.lbl_mel.setMaximumHeight(200)
        self.lbl_mel.setStyleSheet(
            "background:#1a1a2e; border:1px solid #444; border-radius:4px; color:#666;"
        )
        self.lbl_mel.setText("No spectrogram yet")
        self.lbl_mel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        right_col.addWidget(self.lbl_mel)
        text_mel_row.addLayout(right_col, stretch=1)

        outer.addLayout(text_mel_row)

        # ── Pitch / Energy / Duration sliders ────────────────────────────
        h_ctrl = QHBoxLayout()
        for name, default in [("Pitch", 10), ("Energy", 10), ("Duration", 10)]:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(5, 20)
            slider.setValue(default)
            label = QLabel(f"{default/10:.1f}")
            slider.valueChanged.connect(lambda v, lbl=label: lbl.setText(f"{v/10:.1f}"))
            h_ctrl.addWidget(QLabel(name + ":"))
            h_ctrl.addWidget(slider)
            h_ctrl.addWidget(label)
            setattr(self, f"slider_{name.lower()}", slider)

        f_bot = QFormLayout()
        f_bot.addRow(h_ctrl)

        btn_synth = QPushButton("⚡ GENERATE AUDIO")
        btn_synth.setStyleSheet("font-size:16px; padding:12px; background:#4CAF50; color:white;")
        btn_synth.clicked.connect(self.generate_audio)
        f_bot.addRow(btn_synth)

        self.log_syn = QTextEdit(readOnly=True)
        self.log_syn.setMaximumHeight(200)
        f_bot.addRow("Synthesis Log:", self.log_syn)

        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ Play / ⏸ Pause")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        playback_row.addWidget(self.btn_play_pause)

        btn_stop_play = QPushButton("⏹ Stop")
        btn_stop_play.clicked.connect(self.player.stop)
        playback_row.addWidget(btn_stop_play)

        btn_play_last = QPushButton("▶ Play Last Generated")
        btn_play_last.clicked.connect(self.play_last_generated)
        playback_row.addWidget(btn_play_last)
        f_bot.addRow(playback_row)

        outer.addLayout(f_bot)
        g.setLayout(outer)
        tab_lay.addWidget(g)
        tabs.addTab(tab, "Synthesize")

    # ════════════════════════════════════════════════════════════════════════
    # MEL SPECTROGRAM DISPLAY
    # ════════════════════════════════════════════════════════════════════════

    def _display_mel(self, png_path: str):
        """Slot — called from synthesis thread via signal."""
        if not png_path or not os.path.exists(png_path):
            self.lbl_mel.setText("Spectrogram not found")
            return
        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            self.lbl_mel.setText("Could not load spectrogram")
            return
        scaled = pixmap.scaled(
            self.lbl_mel.width(), self.lbl_mel.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_mel.setPixmap(scaled)
        self.lbl_mel.setToolTip(f"Mel spectrogram: {os.path.basename(png_path)}")

    def _clear_mel(self):
        self.lbl_mel.setPixmap(QPixmap())
        self.lbl_mel.setText("No spectrogram yet")

    # ════════════════════════════════════════════════════════════════════════
    # RECORDING
    # ════════════════════════════════════════════════════════════════════════

    def update_rec_label(self, sec):
        self.lbl_rec.setText(f"Recording... {sec:.1f} s")

    def show_rec_error(self, msg):
        QMessageBox.critical(self, "Recording Error", msg)
        self.reset_rec_ui()

    def browse_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac)"
        )
        if path:
            self.selected_audio_file = path
            base = os.path.splitext(os.path.basename(path))[0]
            self.filename_edit.setText(base.replace(" ", "_").lower())
            self.lbl_source.setText(f"Using: {os.path.basename(path)}")
            self.lbl_rec.setText("Ready to generate embedding")
            self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#2e7d32;")
            self.btn_stop.setEnabled(True)
            self.btn_start.setEnabled(False)

    def start_rec(self):
        if self.selected_audio_file:
            QMessageBox.information(self, "File selected",
                "An audio file is already selected.\nClick Stop & Generate to process it.")
            return
        self.audio_chunks = []
        self.recording = True
        self.start_t = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_rec.setText("Recording... 0.0 s")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#d32f2f;")
        self.record_thread = threading.Thread(target=self._rec_worker, daemon=True)
        self.record_thread.start()

    def _rec_worker(self):
        def cb(indata, frames, ti, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                dtype='float32', callback=cb):
                while self.recording:
                    self.rec_signals.update_time.emit(time.time() - self.start_t)
                    sd.sleep(150)
        except Exception as e:
            self.rec_signals.error_message.emit(str(e))

    def stop_rec(self):
        self.recording = False
        self.lbl_rec.setText("Processing...")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=5)

        sid = next_global_id()
        raw_path = os.path.join("createdataset", f"{sid}.wav")

        if self.selected_audio_file:
            try:
                audio, sr = torchaudio.load(self.selected_audio_file)
                if sr != self.sample_rate:
                    audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
                if audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                torchaudio.save(raw_path, audio, self.sample_rate)
            except Exception as e:
                QMessageBox.critical(self, "File Error", str(e))
                self.reset_rec_ui()
                return
        else:
            if not self.audio_chunks:
                self.reset_rec_ui()
                return
            arr = np.concatenate(self.audio_chunks, axis=0)
            torchaudio.save(raw_path, torch.from_numpy(arr.T).float(), self.sample_rate)

        try:
            pw, pt, _ = full_pipeline(raw_path, sid, output_dir=UNSEEN_DIR)
            name = self.filename_edit.text().strip()
            if name:
                with open(os.path.join(UNSEEN_DIR, f"{sid}.txt"), "w", encoding="utf-8") as nf:
                    nf.write(name)
            QMessageBox.information(self, "Success",
                f"Unseen speaker created!\n\n"
                f"ID:        {sid}\n"
                f"Raw:       {raw_path}\n"
                f"Processed: {pw}\n"
                f"Embedding: {pt}")
            self.refresh_speakers()
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", str(e))

        self.reset_rec_ui()
        self.selected_audio_file = None
        self.lbl_source.setText("No source selected")
        self.btn_start.setEnabled(True)

    def reset_rec_ui(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_rec.setText("Ready")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")
        self.audio_chunks = []

    # ════════════════════════════════════════════════════════════════════════
    # SPEAKER LISTS
    # ════════════════════════════════════════════════════════════════════════

    def _list_sids(self, directory):
        """
        For SEEN_DIR: list speakers that have a .ecapa_averaged_embedding
                      (fallback: .pt if no averaged embedding exists)
        For UNSEEN_DIR: list speakers that have a .pt
        """
        if not os.path.isdir(directory):
            return []
        sids = set()
        if directory == SEEN_DIR:
            # for f in os.listdir(directory):
            #     if f.endswith(".ecapa_averaged_embedding"):
            #         stem = f[: -len(".ecapa_averaged_embedding")]
            #         sids.add(stem)
            # # Also show any seen speakers that only have .pt (no averaged embedding yet)
            for f in os.listdir(directory):
                stem = os.path.splitext(f)[0]
                if f.endswith(".pt") and not stem.endswith("_processed"):
                    sids.add(stem)
        else:
            for f in os.listdir(directory):
                stem = os.path.splitext(f)[0]
                if f.endswith(".pt") and not stem.endswith("_processed"):
                    sids.add(stem)

        sids = list(sids)
        sids.sort(key=lambda x: int(x) if x.isdigit() else float("inf"))
        return sids

    def _emb_type_label(self, directory, sid):
        """Return a short string showing which embedding file will be used."""
        if directory == SEEN_DIR:
            p = os.path.join(directory, f"{sid}.ecapa_averaged_embedding")
            return "avg_emb" if os.path.exists(p) else ".pt"
        return ".pt"

    def refresh_speakers(self):
        self.lst_seen.clear()
        for sid in self._list_sids(SEEN_DIR):
            note  = self._get_note(sid, SEEN_DIR)
            etype = self._emb_type_label(SEEN_DIR, sid)
            txt   = f"Speaker {sid}   •   [{etype}]   •   {sid}_processed.wav"
            if note:
                txt += f"     [{note}]"
            self.lst_seen.addItem(txt)

        self.lst_unseen.clear()
        for sid in self._list_sids(UNSEEN_DIR):
            note = self._get_note(sid, UNSEEN_DIR)
            txt  = f"Speaker {sid}   •   [.pt]   •   {sid}_processed.wav"
            if note:
                txt += f"     [{note}]"
            self.lst_unseen.addItem(txt)

        self._refresh_speaker_combo()

    def _refresh_speaker_combo(self):
        directory = self.cmb_set.currentData()
        self.cmb_speaker.clear()
        for sid in self._list_sids(directory):
            note  = self._get_note(sid, directory)
            etype = self._emb_type_label(directory, sid)
            label = f"{sid} [{etype}]"
            if note:
                label += f" — {note}"
            self.cmb_speaker.addItem(label, (directory, sid))

    def _on_set_changed(self):
        self._refresh_speaker_combo()

    def _sid_from_item(self, item):
        return item.text().split()[1]

    def _get_note(self, sid, directory):
        p = os.path.join(directory, f"{sid}.txt")
        if os.path.exists(p):
            try:
                return open(p, encoding="utf-8").read().strip()
            except Exception:
                pass
        return ""

    def _on_sel_changed(self, lst, directory):
        item = lst.currentItem()
        if not item:
            self.txt_note.clear()
            self.btn_pause_listen.setEnabled(False)
            return
        sid = self._sid_from_item(item)
        self.txt_note.setPlainText(self._get_note(sid, directory))
        self._active_note_sid = sid
        self._active_note_dir = directory

    def _on_dblclick(self, item, directory):
        sid = self._sid_from_item(item)
        self._show_info(sid, directory)
        self.txt_note.setPlainText(self._get_note(sid, directory))
        self._active_note_sid = sid
        self._active_note_dir = directory

    def _show_info(self, sid, directory):
        try:
            emb, emb_path = load_embedding(directory, sid)
        except FileNotFoundError:
            self.txt_details.setHtml(f"<b>No embedding found for speaker {sid}</b>")
            return

        note  = self._get_note(sid, directory)
        label = "seen" if directory == SEEN_DIR else "unseen"
        efile = os.path.basename(emb_path)
        info  = (
            f"<b>Speaker ID:</b> {sid}   <b>Set:</b> {label}<br>"
            f"<b>Embedding file:</b> {efile}<br>"
        )
        if note:
            info += f"<b>Note:</b> {note}<br><br>"
        info += (
            f"<b>Shape:</b> {emb.shape}<br>"
            f"<b>min/max:</b> {emb.min():.6f} / {emb.max():.6f}<br>"
            f"<b>L2 norm:</b> {emb.norm():.6f}<br><br>"
            f"First 20 values: {emb.view(-1)[:20].tolist()}"
        )
        self.txt_details.setHtml(info)
        self._play_processed(sid, directory)

    def _play_processed(self, sid, directory):
        p = os.path.join(directory, f"{sid}_processed.wav")
        if os.path.exists(p):
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(os.path.abspath(p))))
            self.player.play()
            self.btn_pause_listen.setText("⏸ Pause Listening")
            self.btn_pause_listen.setEnabled(True)
        else:
            self.btn_pause_listen.setEnabled(False)

    def _play_selected(self, lst, directory):
        item = lst.currentItem()
        if item:
            self._play_processed(self._sid_from_item(item), directory)

    def _delete_unseen(self):
        item = self.lst_unseen.currentItem()
        if not item:
            QMessageBox.warning(self, "No selection", "Select an unseen speaker to delete.")
            return
        sid = self._sid_from_item(item)
        if QMessageBox.question(self, "Confirm", f"Delete speaker {sid}?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        for suffix in [".pt", "_processed.wav", ".txt", ".wav"]:
            p = os.path.join(UNSEEN_DIR, f"{sid}{suffix}")
            if os.path.exists(p):
                os.remove(p)
        self.refresh_speakers()

    def save_speaker_note(self):
        if not hasattr(self, "_active_note_sid"):
            QMessageBox.warning(self, "No speaker", "Select a speaker first.")
            return
        sid       = self._active_note_sid
        directory = self._active_note_dir
        note_path = os.path.join(directory, f"{sid}.txt")
        text      = self.txt_note.toPlainText().strip()
        if not text:
            if os.path.exists(note_path):
                os.remove(note_path)
            self.refresh_speakers()
            return
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(text)
        QMessageBox.information(self, "Saved", f"Note saved for speaker {sid}")
        self.refresh_speakers()

    # ════════════════════════════════════════════════════════════════════════
    # PLAYBACK
    # ════════════════════════════════════════════════════════════════════════

    def toggle_play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play_pause.setText("▶ Resume")
        else:
            self.player.play()
            self.btn_play_pause.setText("⏸ Pause")

    def toggle_pause_listen(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_pause_listen.setText("▶ Resume Listening")
        else:
            self.player.play()
            self.btn_pause_listen.setText("⏸ Pause Listening")

    def play_last_generated(self):
        if not self.last_synthesized_file or not os.path.exists(self.last_synthesized_file):
            QMessageBox.information(self, "No file", "No recent synthesis found.")
            return
        self.player.setMedia(
            QMediaContent(QUrl.fromLocalFile(os.path.abspath(self.last_synthesized_file))))
        self.player.play()
        self.btn_play_pause.setText("⏸ Pause")
        self.status.showMessage(f"Playing: {os.path.basename(self.last_synthesized_file)}")

    # ════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ════════════════════════════════════════════════════════════════════════

    def generate_audio(self):
        idx = self.cmb_speaker.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Missing input", "No speaker available.")
            return
        directory, sid = self.cmb_speaker.itemData(idx)
        text = self.txt_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing input", "Enter text to synthesize.")
            return

        # Resolve the correct embedding path
        emb_path = get_embedding_path(directory, sid)
        if not os.path.exists(emb_path):
            QMessageBox.critical(self, "Missing embedding",
                f"Could not find embedding for speaker {sid}:\n{emb_path}")
            return

        p_val = self.slider_pitch.value()    / 10.0
        e_val = self.slider_energy.value()   / 10.0
        d_val = self.slider_duration.value() / 10.0

        self.log_syn.clear()
        self.log_syn.append(
            f"Synthesizing with speaker {sid} ...\n"
            f"  Embedding: {os.path.basename(emb_path)}"
        )

        cmd = [
            sys.executable, "generate.py",
            "--restore_step", "626000",
            "--mode", "single",
            "--text", text,
            "--speaker_emb", emb_path,          # ← correct file for seen/unseen
            "-p", "config/LibriTTS/preprocess.yaml",
            "-m", "config/LibriTTS/model.yaml",
            "-t", "config/LibriTTS/train.yaml",
            "--pitch_control",    f"{p_val:.2f}",
            "--energy_control",   f"{e_val:.2f}",
            "--duration_control", f"{d_val:.2f}",
        ]

        threading.Thread(
            target=self._run_synth,
            args=(cmd, directory, sid),
            daemon=True
        ).start()

    def _run_synth(self, cmd, speaker_dir, speaker_sid):
        out_dir = "output/result/LibriTTS"
        os.makedirs(out_dir, exist_ok=True)

        # Snapshot timestamps of ALL existing wavs and pngs before synthesis
        before_wav = {}
        before_png = {}
        for f in glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True):
            before_wav[f] = os.path.getmtime(f)
        for f in glob.glob(os.path.join(out_dir, "**", "*.png"), recursive=True):
            before_png[f] = os.path.getmtime(f)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.stdout.strip():
                self.log_syn.append(result.stdout)
            if result.stderr.strip():
                self.log_syn.append(result.stderr)

            time.sleep(1.5)

            # ── Detect new wav ────────────────────────────────────────────
            after_wavs = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
            new_wavs   = [f for f in after_wavs
                          if f not in before_wav or os.path.getmtime(f) > before_wav.get(f, 0)]

            if new_wavs:
                newest_wav = max(new_wavs, key=os.path.getmtime)
            else:
                all_wavs = glob.glob(os.path.join(out_dir, "**", "*.wav"), recursive=True)
                newest_wav = max(all_wavs, key=os.path.getmtime) if all_wavs else None

            if newest_wav:
                self.last_synthesized_file = newest_wav
                self.log_syn.append(f"\n✓ Generated: {os.path.basename(newest_wav)}")
                self.log_syn.append(f"  Path: {newest_wav}")
                self._log_similarity(speaker_dir, speaker_sid, newest_wav)

                # ── Detect mel PNG ────────────────────────────────────────
                after_pngs = glob.glob(os.path.join(out_dir, "**", "*.png"), recursive=True)
                new_pngs   = [f for f in after_pngs
                              if f not in before_png or os.path.getmtime(f) > before_png.get(f, 0)]

                mel_png = None
                if new_pngs:
                    mel_png = max(new_pngs, key=os.path.getmtime)
                else:
                    # Try same stem as wav
                    stem    = os.path.splitext(newest_wav)[0]
                    if os.path.exists(stem + ".png"):
                        mel_png = stem + ".png"

                if mel_png:
                    self.log_syn.append(f"  Spectrogram: {os.path.basename(mel_png)}")
                    self.rec_signals.mel_ready.emit(mel_png)
                else:
                    self.log_syn.append("  ⚠ No mel spectrogram PNG found.")
            else:
                self.log_syn.append(
                    "\n⚠ No .wav file found under output/result/LibriTTS/\n"
                    "  Check that synthesize6.py writes to that directory."
                )

            if result.returncode != 0:
                self.log_syn.append("\n✗ Synthesis process returned a non-zero exit code.")

        except subprocess.TimeoutExpired:
            self.log_syn.append("\n✗ Synthesis timed out after 5 minutes.")
        except Exception as ex:
            self.log_syn.append(f"\n✗ Exception: {ex}")

    def _log_similarity(self, speaker_dir, speaker_sid, generated_wav):
        try:
            try:
                original_emb, _ = load_embedding(speaker_dir, speaker_sid)
            except FileNotFoundError:
                self.log_syn.append("\n⚠ Speaker embedding not found — skipping similarity.")
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            audio, sr = torchaudio.load(generated_wav)
            if audio.ndim > 1 and audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if sr != 16000:
                audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            audio = trim_silence(audio)
            audio = audio.squeeze(0).to(device).float()

            model = get_ECAPA_TDNN_MODEL(device=device)
            model.eval()
            with torch.no_grad():
                gen_emb = speaker_embedding_extractor(model, audio.unsqueeze(0))
            gen_emb = torch.nn.functional.normalize(gen_emb, p=2, dim=1)

            pct = cosine_similarity_pct(original_emb, gen_emb)

            self.log_syn.append(
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f" Speaker Similarity: {pct:.2f}%\n"
                f"   (ECAPA-TDNN cosine similarity —\n"
                f"    original speaker {speaker_sid} vs generated audio)\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

        except Exception as e:
            self.log_syn.append(f"\n⚠ Could not compute similarity: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TTSApp()
    window.show()
    sys.exit(app.exec_())
