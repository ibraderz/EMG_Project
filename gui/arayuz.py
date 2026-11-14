import sys
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                             QFileDialog, QComboBox, QGridLayout, QFrame, QStackedWidget)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont
import pyqtgraph as pg
from scipy.signal import butter, lfilter


# ==========================================
# 1. MATEMATİK VE SİNYAL İŞLEME
# ==========================================
class SignalProcessor:
    @staticmethod
    def apply_filter(data, cutoff, fs, btype='low', order=4):
        if len(data) < 10:
            return data
        nyq = 0.5 * fs
        normal_cutoff = np.array(cutoff) / nyq
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return lfilter(b, a, data)

    @staticmethod
    def get_fft(data, fs):
        n = len(data)
        if n < 2:
            return np.array([]), np.array([])
        freqs = np.fft.rfftfreq(n, d=1/fs)
        mags = np.abs(np.fft.rfft(data))
        return freqs, mags


# ==========================================
# 2. ROBOT KOL ÇİZİM BİLEŞENİ
# ==========================================
class RobotArmWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(350)
        self.is_clenched = False
        self.rms_val = 0.0

    def update_state(self, is_clenched, rms):
        self.is_clenched = is_clenched
        self.rms_val = rms
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        
        cx = self.width() // 2
        arm_color = QColor("#d4a373")

        painter.setBrush(QBrush(arm_color))
        painter.drawRect(cx - 40, 50, 80, 160)
        painter.setPen(QColor("gray"))
        painter.drawText(cx + 50, 130, "Ust Kol")

        painter.setBrush(QBrush(QColor("#7f8c8d")))
        painter.drawEllipse(cx - 15, 200, 30, 20)

        painter.setBrush(QBrush(arm_color))
        painter.drawRect(cx - 35, 215, 70, 180)
        painter.drawText(cx + 45, 300, "On Kol")

        hand_color = QColor("#e74c3c") if self.is_clenched else QColor("#3498db")
        painter.setBrush(QBrush(hand_color))
        painter.drawRoundedRect(cx - 30, 400, 60, 50, 8, 8)
        painter.drawText(cx + 40, 430, "El")

        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        painter.setPen(QColor("#e74c3c"))
        status = "KAPALI" if self.is_clenched else "ACIK"
        painter.drawText(20, self.height() - 60, f"EL: {status}")
        painter.setPen(QColor("#3498db"))
        painter.drawText(20, self.height() - 25, f"KOL RMS: {self.rms_val:.2f}")


# ==========================================
# 3. SAYFALAR (SÜZGEÇ VE PROTEZ)
# ==========================================
class SuzgecPage(QWidget):
    def __init__(self):
        super().__init__()
        self.fft_aktif = False
        self.layout_kilitli = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_geri = QPushButton("◀ Ana Ekrana Dön")
        self.btn_geri.setStyleSheet("background-color: #555; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        top_bar.addWidget(self.btn_geri)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.grid = QGridLayout()
        self.p_raw = pg.PlotWidget(title="Ham EMG Sinyali"); self.c_raw = self.p_raw.plot(pen='#2ecc71')
        self.p_lp = pg.PlotWidget(title="Alcak Geciren Filtre"); self.c_lp = self.p_lp.plot(pen='#f39c12')
        self.p_hp = pg.PlotWidget(title="Yuksek Geciren Filtre"); self.c_hp = self.p_hp.plot(pen='#e74c3c')
        self.p_bp = pg.PlotWidget(title="Band Geciren Filtre"); self.c_bp = self.p_bp.plot(pen='#3498db')
        
        self.f_raw = pg.PlotWidget(title="FFT - Ham Sinyal"); self.cf_raw = self.f_raw.plot(pen='#2ecc71')
        self.f_lp = pg.PlotWidget(title="FFT - Alcak Geciren"); self.cf_lp = self.f_lp.plot(pen='#f39c12')
        self.f_hp = pg.PlotWidget(title="FFT - Yuksek Geciren"); self.cf_hp = self.f_hp.plot(pen='#e74c3c')
        self.f_bp = pg.PlotWidget(title="FFT - Band Geciren"); self.cf_bp = self.f_bp.plot(pen='#3498db')

        self.fft_grafikleri = [self.f_raw, self.f_lp, self.f_hp, self.f_bp]
        plots = [(self.p_raw, self.f_raw, 0), (self.p_lp, self.f_lp, 1), 
                 (self.p_hp, self.f_hp, 2), (self.p_bp, self.f_bp, 3)]
        
        for p_t, p_f, row in plots:
            p_t.setBackground('k')
            p_f.setBackground('k')
            self.grid.addWidget(p_t, row, 0)
            self.grid.addWidget(p_f, row, 1)
            p_f.setVisible(False)
            
        layout.addLayout(self.grid)

        ctrl = QHBoxLayout()
        self.txt_ag, self.txt_yg = QLineEdit("5"), QLineEdit("20")
        self.txt_bg_l, self.txt_bg_h = QLineEdit("50"), QLineEdit("80")
        
        self.txt_ag.setFixedWidth(40)
        self.txt_yg.setFixedWidth(40)
        self.txt_bg_l.setFixedWidth(40)
        self.txt_bg_h.setFixedWidth(40)

        self.btn_uygula = QPushButton("Uygula")
        self.btn_uygula.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 6px;")
        self.btn_uygula.clicked.connect(
            lambda: self.update_plots(self.window().page_main.buffer, self.window().page_main.fs)
        )
        
        self.btn_fft = QPushButton("FFT (Pasif)")
        self.btn_fft.setStyleSheet("background-color: #555; color: white; font-weight: bold; padding: 6px;")
        self.btn_fft.clicked.connect(self.toggle_fft)

        self.btn_norm = QPushButton("Norm (Pasif)")
        self.btn_norm.setStyleSheet("background-color: #555; color: white; font-weight: bold; padding: 6px;")

        ctrl.addWidget(QLabel("AG (Hz):")); ctrl.addWidget(self.txt_ag)
        ctrl.addWidget(QLabel("YG (Hz):")); ctrl.addWidget(self.txt_yg)
        ctrl.addWidget(QLabel("BG (Hz):")); ctrl.addWidget(self.txt_bg_l); ctrl.addWidget(QLabel("-")); ctrl.addWidget(self.txt_bg_h)
        ctrl.addWidget(self.btn_uygula)
        ctrl.addWidget(self.btn_fft)
        ctrl.addWidget(self.btn_norm)
        ctrl.addStretch()
        layout.addLayout(ctrl)

    def toggle_fft(self):
        self.layout_kilitli = True
        self.fft_aktif = not self.fft_aktif
        
        if not self.fft_aktif:
            self.cf_raw.setData([], [])
            self.cf_lp.setData([], [])
            self.cf_hp.setData([], [])
            self.cf_bp.setData([], [])

        for g in self.fft_grafikleri:
            g.setVisible(self.fft_aktif)
            
        if self.fft_aktif:
            self.btn_fft.setText("FFT (Aktif)")
            self.btn_fft.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 6px;")
        else:
            self.btn_fft.setText("FFT (Pasif)")
            self.btn_fft.setStyleSheet("background-color: #555; color: white; font-weight: bold; padding: 6px;")

        QApplication.processEvents()
        self.layout_kilitli = False

    def update_plots(self, data, fs):
        if getattr(self, 'layout_kilitli', False):
            return

        try:
            ag_fc, yg_fc = float(self.txt_ag.text()), float(self.txt_yg.text())
            bg_fc = [float(self.txt_bg_l.text()), float(self.txt_bg_h.text())]

            lp_data = SignalProcessor.apply_filter(data, ag_fc, fs, 'low')
            hp_data = SignalProcessor.apply_filter(data, yg_fc, fs, 'high')
            bp_data = SignalProcessor.apply_filter(data, bg_fc, fs, 'band')

            self.c_raw.setData(data)
            self.c_lp.setData(lp_data)
            self.c_hp.setData(hp_data)
            self.c_bp.setData(bp_data)

            if self.fft_aktif:
                for plot_curve, sig in [(self.cf_raw, data), (self.cf_lp, lp_data), 
                                        (self.cf_hp, hp_data), (self.cf_bp, bp_data)]:
                    f, m = SignalProcessor.get_fft(sig, fs)
                    plot_curve.setData(f, m)
        except Exception:
            pass


class ProtezPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_geri = QPushButton("◀ Ana Ekrana Dön")
        self.btn_geri.setStyleSheet("background-color: #555; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        top_bar.addWidget(self.btn_geri)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        main_split = QHBoxLayout()
        self.plot = pg.PlotWidget(title="Sinyal (yesil) + RMS (turuncu)")
        self.plot.setBackground('k')
        self.curve_sig = self.plot.plot(pen='#2ecc71')
        self.curve_rms = self.plot.plot(pen=pg.mkPen('#e67e22', width=2))
        self.line_th = pg.InfiniteLine(angle=0, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.line_th)
        main_split.addWidget(self.plot, stretch=2)

        self.arm = RobotArmWidget()
        main_split.addWidget(self.arm, stretch=1)
        layout.addLayout(main_split)

        ctrl = QHBoxLayout()
        self.txt_esik = QLineEdit("300")
        self.line_th.setValue(300)
        ctrl.addWidget(QLabel("Esik (V):"))
        ctrl.addWidget(self.txt_esik)
        ctrl.addStretch()
        layout.addLayout(ctrl)

    def update_plots(self, data, fs):
        try:
            rms = np.sqrt(np.mean(data[-100:]**2))
            esik = float(self.txt_esik.text())
            self.line_th.setValue(esik)
            
            self.curve_sig.setData(data)
            self.curve_rms.setData(np.full(len(data), rms))
            self.arm.update_state(rms > esik, rms)
        except Exception:
            pass


# ==========================================
# 4. ANA SAYFA VE SİSTEM YÖNETİCİSİ
# ==========================================
class AnaEkranPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window 
        
        self.fs = 1000
        self.buffer = np.zeros(1000)
        self.serial_conn = None
        self.is_running = False

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_data)

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.plot = pg.PlotWidget(title="Ham EMG Sinyali")
        self.plot.setBackground('#1a1a1a')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot(pen='#00ff88')
        layout.addWidget(self.plot)

        status_layout = QHBoxLayout()
        self.lbl_voltaj = QLabel("Voltaj: 0.000 V")
        self.lbl_voltaj.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 16px;")
        self.lbl_durum = QLabel("Durum: Durduruldu")
        self.lbl_durum.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.lbl_voltaj)
        status_layout.addWidget(self.lbl_durum)
        layout.addLayout(status_layout)

        port_layout = QHBoxLayout()
        self.cmb_port = QComboBox()
        self.cmb_port.setMinimumWidth(250)
        self.refresh_ports()
        
        self.btn_tara = QPushButton("Portlari Tara")
        self.btn_tara.setStyleSheet("background-color: #557b9d; color: white;")
        self.btn_tara.clicked.connect(self.refresh_ports)
        
        self.txt_manuel = QLineEdit()
        self.txt_manuel.setPlaceholderText("Orn: COM3")
        
        self.btn_baglan = QPushButton("Baglan")
        self.btn_baglan.setStyleSheet("background-color: #557b9d; color: white;")
        self.btn_baglan.clicked.connect(self.baglan)

        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.cmb_port)
        port_layout.addWidget(self.btn_tara)
        port_layout.addWidget(QLabel("Manuel:"))
        port_layout.addWidget(self.txt_manuel)
        port_layout.addWidget(self.btn_baglan)
        port_layout.addStretch()
        layout.addLayout(port_layout)

        btn_layout = QHBoxLayout()
        
        btn_data = [
            ("Baslat", "#2ecc71", self.baslat),
            ("Durdur", "#e74c3c", self.durdur),
            ("Suzgec", "#f39c12", lambda: self.main_window.stacked.setCurrentIndex(1)),
            ("Analiz", "#3498db", lambda: self.main_window.stacked.setCurrentIndex(1)),
            ("Protez Kol", "#e84393", lambda: self.main_window.stacked.setCurrentIndex(2)),
            ("Normalizasyon", "#1abc9c", self.normalizasyon_uygula),
            ("Dosyaya Kaydet", "#5dade2", self.kaydet),
            ("Dosya Ac", "#9b59b6", self.yukle),
            ("Temizle", "#34495e", self.temizle)
        ]

        for text, color, func in btn_data:
            btn = QPushButton(text)
            btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 10px; border-radius: 3px;")
            btn.clicked.connect(func)
            btn_layout.addWidget(btn)
            
        layout.addLayout(btn_layout)

    def refresh_ports(self):
        self.cmb_port.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.cmb_port.addItems(ports)

    def baglan(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.serial_conn = None
            self.btn_baglan.setText("Baglan")
            self.lbl_durum.setText("Durum: Baglanti Kesildi")
        else:
            port = self.txt_manuel.text() if self.txt_manuel.text() else self.cmb_port.currentText()
            if port:
                try:
                    self.serial_conn = serial.Serial(port, 115200, timeout=0.01)
                    self.btn_baglan.setText("Kes")
                    self.lbl_durum.setText(f"Durum: {port} Baglandi")
                except Exception as e:
                    self.lbl_durum.setText(f"Hata: {e}")

    def baslat(self):
        self.is_running = True
        self.timer.start(10)
        self.lbl_durum.setText("Durum: Calisiyor")

    def durdur(self):
        self.is_running = False
        self.timer.stop()
        self.lbl_durum.setText("Durum: Durduruldu")

    def read_data(self):
        veri_geldi_mi = False
        val = 0.0

        if self.serial_conn and self.serial_conn.is_open:
            try:
                while self.serial_conn.in_waiting > 0:
                    raw_data = self.serial_conn.readline()
                    line = raw_data.decode('utf-8').strip()
                    if line:
                        val = float(line)
                        self.buffer = np.roll(self.buffer, -1)
                        self.buffer[-1] = val
                        veri_geldi_mi = True
            except OSError:
                self.serial_conn.close()
                self.serial_conn = None
                self.btn_baglan.setText("Baglan")
                self.lbl_durum.setText("Durum: Port Koptu!")
                self.durdur()
                return
            except ValueError:
                pass
        elif not self.serial_conn:
            val = np.random.normal(0, 0.2)
            if np.random.random() > 0.95:
                val += 2.0
            self.buffer = np.roll(self.buffer, -1)
            self.buffer[-1] = val
            veri_geldi_mi = True

        if veri_geldi_mi:
            self.lbl_voltaj.setText(f"Voltaj: {val:.3f} V")
            self.curve.setData(self.buffer)

            current_page = self.main_window.stacked.currentWidget()
            if current_page == self.main_window.page_suzgec:
                self.main_window.page_suzgec.update_plots(self.buffer, self.fs)
            elif current_page == self.main_window.page_protez:
                self.main_window.page_protez.update_plots(self.buffer, self.fs)

    def kaydet(self):
        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "CSV Files (*.csv)")
        if path:
            pd.DataFrame(self.buffer).to_csv(path, index=False)

    def yukle(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ac", "", "CSV Files (*.csv)")
        if path: 
            self.buffer = pd.read_csv(path).iloc[:, 0].values
            self.curve.setData(self.buffer)
            self.lbl_durum.setText("Durum: Dosya Yuklendi")
            self.main_window.page_suzgec.update_plots(self.buffer, self.fs)
            self.main_window.page_protez.update_plots(self.buffer, self.fs)

    def temizle(self):
        self.buffer = np.zeros(1000)
        self.curve.setData(self.buffer)
        self.lbl_durum.setText("Durum: Temizlendi")
        
    def normalizasyon_uygula(self):
        max_val = np.max(self.buffer)
        min_val = np.min(self.buffer)
        if max_val != min_val:
            self.buffer = (self.buffer - min_val) / (max_val - min_val)
            self.curve.setData(self.buffer)
            self.lbl_durum.setText("Durum: Normalize Edildi")


# ==========================================
# 5. UYGULAMA PENCERESİ VE GEÇİŞLER
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMG Sinyal Arayuzu - Doc.Dr.Cengiz TEPE")
        self.resize(1300, 800)

        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)

        self.page_main = AnaEkranPage(self)
        self.page_suzgec = SuzgecPage()
        self.page_protez = ProtezPage()

        self.stacked.addWidget(self.page_main)
        self.stacked.addWidget(self.page_suzgec)
        self.stacked.addWidget(self.page_protez)

        self.page_suzgec.btn_geri.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.page_protez.btn_geri.clicked.connect(lambda: self.stacked.setCurrentIndex(0))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())