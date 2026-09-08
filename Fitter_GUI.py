import os
import glob
import re
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.optimize import minimize
import emcee

try:
    import corner
except ImportError:
    corner = None

# Conversion constant: converts erg/s/cm^2/A * micron^2 to uJy
FLUX_CONV_FACTOR = 3.33564095e18

# ----------------- Physics & Model Helpers -----------------

def break_at_gaps(x, y, gap_factor=5):
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.diff(x)
    median_dx = np.nanmedian(dx)
    gap_idx = np.where(dx > gap_factor * median_dx)[0]
    
    x_new, y_new = [], []
    for i in range(len(x) - 1):
        x_new.append(x[i])
        y_new.append(y[i])
        if i in gap_idx:
            x_new.append(np.nan)
            y_new.append(np.nan)
            
    x_new.append(x[-1])
    y_new.append(y[-1])
    return np.array(x_new), np.array(y_new)

def load_extinction_law(csv_path, obs_wave):
    df_ext = pd.read_csv(csv_path).sort_values('wavelength')
    law_wave = df_ext['wavelength'].values
    law_ratio = df_ext['Ak'].values 
    extinction_curve = np.interp(obs_wave, law_wave, law_ratio, left=np.nan, right=np.nan)
    return np.nan_to_num(extinction_curve, nan=0.0)

def apply_extinction(flux, extinction_curve, Ak):
    A_lambda = extinction_curve * Ak
    return flux * 10**(-0.4 * A_lambda)

def load_models(model_folder, spec_wave, filename_pattern=r'lte(\d{2,3})'):
    model_files = sorted([f for f in os.listdir(model_folder) if f.endswith('.csv')])
    temps, flux_on_fit, filenames = [], [], []

    for fname in model_files:
        m = re.search(filename_pattern, fname)
        temp_k = int(m.group(1)) * 100 if m else 0
        df = pd.read_csv(os.path.join(model_folder, fname))
        mod_wave, mod_flux = df['wavelength_micron'].values, df['flux'].values
        
        mod_flux_uJy = mod_flux * (mod_wave ** 2) * FLUX_CONV_FACTOR
        flux_spec = np.interp(spec_wave, mod_wave, mod_flux_uJy, left=0, right=0)
        
        temps.append(temp_k)
        flux_on_fit.append(flux_spec)
        filenames.append(fname)

    temps = np.array(temps)
    sort_idx = np.argsort(temps)
    return temps[sort_idx], np.array(flux_on_fit)[sort_idx], np.array(filenames)[sort_idx]

# ----------------- GUI Application -----------------

class SpectrumFitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Stellar Spectrum MCMC Fitter")
        self.root.geometry("1300x900")

        # Configuration Paths with requested defaults
        self.obs_file_path = tk.StringVar()
        self.model_folder_path = tk.StringVar(value='bt-settl-agss/R1000')
        self.extinction_path = tk.StringVar(value='KP5_extinction_curve.csv')
        self.output_folder_path = tk.StringVar(value='fits_output')

        # Fit Parameters
        self.teff_val = tk.DoubleVar(value=4000)
        self.ak_val = tk.DoubleVar(value=1.0)
        self.log_scale_val = tk.DoubleVar(value=-5.0)

        # Loaded Cache Data
        self.df_obs_full = None
        self.obs_wave = None
        self.obs_flux = None
        self.obs_err = None
        self.fit_weights = None
        self.temps = None
        self.flux_on_fit = None
        self.filenames = None
        self.extinction_curve_fit = None

        self._build_ui()

    def _build_ui(self):
        # 1. Path Selection Panel
        path_frame = ttk.LabelFrame(self.root, text=" 1. Data & Model Paths ", padding=10)
        path_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(path_frame, text="Observed File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.obs_file_path, width=55).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Select File", command=self.browse_obs_file).grid(row=0, column=2, padx=2)

        ttk.Label(path_frame, text="Models Folder:").grid(row=1, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.model_folder_path, width=55).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Select Folder", command=self.browse_model_folder).grid(row=1, column=2, padx=2)

        ttk.Label(path_frame, text="Extinction CSV:").grid(row=2, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.extinction_path, width=55).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Select File", command=self.browse_extinction_file).grid(row=2, column=2, padx=2)

        ttk.Label(path_frame, text="Output Folder:").grid(row=3, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.output_folder_path, width=55).grid(row=3, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Select Folder", command=self.browse_output_folder).grid(row=3, column=2, padx=2)

        ttk.Button(path_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, rowspan=4, padx=15, sticky="ns")

        # Main Workspace Split
        workspace = ttk.Frame(self.root)
        workspace.pack(fill="both", expand=True, padx=10, pady=5)

        # 2. Controls Panel (Sliders)
        ctrl_frame = ttk.LabelFrame(workspace, text=" 2. Parameter Controls ", padding=10)
        ctrl_frame.pack(side="left", fill="y", padx=(0, 5))

        # T_eff Slider
        ttk.Label(ctrl_frame, text="T_eff (K):").pack(anchor="w", pady=(5, 0))
        self.teff_slider = ttk.Scale(
            ctrl_frame, from_=2000, to=10000, variable=self.teff_val,
            orient="horizontal", command=self.on_slider_move, length=240
        )
        self.teff_slider.pack(fill="x")
        self.teff_label = ttk.Label(ctrl_frame, text="4000 K")
        self.teff_label.pack(anchor="e")

        # A_K Slider
        ttk.Label(ctrl_frame, text="A_K (mag):").pack(anchor="w", pady=(10, 0))
        self.ak_slider = ttk.Scale(
            ctrl_frame, from_=0.0, to=30.0, variable=self.ak_val,
            orient="horizontal", command=self.on_slider_move, length=240
        )
        self.ak_slider.pack(fill="x")
        self.ak_label = ttk.Label(ctrl_frame, text="1.00 mag")
        self.ak_label.pack(anchor="e")

        # ln(Scale) Slider
        ttk.Label(ctrl_frame, text="ln(Scale Factor):").pack(anchor="w", pady=(10, 0))
        self.scale_slider = ttk.Scale(
            ctrl_frame, from_=-50.0, to=20.0, variable=self.log_scale_val,
            orient="horizontal", command=self.on_slider_move, length=240
        )
        self.scale_slider.pack(fill="x")
        self.scale_label = ttk.Label(ctrl_frame, text="-5.00 (scale: 6.74e-03)")
        self.scale_label.pack(anchor="e")

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=15)

        self.btn_mcmc = ttk.Button(ctrl_frame, text="Run MCMC Fit (Current)", command=self.start_mcmc_thread)
        self.btn_mcmc.pack(fill="x", pady=5)

        self.status_label = ttk.Label(ctrl_frame, text="Status: Ready", wraplength=220, foreground="blue")
        self.status_label.pack(anchor="w", pady=10)

        # 3. Plot Preview Canvas
        plot_frame = ttk.LabelFrame(workspace, text=" Interactive Spectrum Plot ", padding=5)
        plot_frame.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

    # ----------------- Browsers -----------------

    def browse_obs_file(self):
        f = filedialog.askopenfilename(filetypes=[("CSV/TXT files", "*.csv *.txt"), ("All files", "*.*")])
        if f:
            self.obs_file_path.set(f)

    def browse_model_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.model_folder_path.set(d)

    def browse_extinction_file(self):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if f:
            self.extinction_path.set(f)

    def browse_output_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.output_folder_path.set(d)

    # ----------------- Data Loading & Processing -----------------

    def load_data(self):
        try:
            obs_csv = self.obs_file_path.get()
            model_folder = self.model_folder_path.get()
            extinction_csv = self.extinction_path.get()

            if not os.path.exists(obs_csv):
                raise FileNotFoundError("Observational file not found.")
            if not os.path.exists(model_folder):
                raise FileNotFoundError(f"Model directory '{model_folder}' not found.")
            if not os.path.exists(extinction_csv):
                raise FileNotFoundError(f"Extinction CSV '{extinction_csv}' not found.")

            self.df_obs_full = pd.read_csv(obs_csv, sep=None, engine='python')
            if len(self.df_obs_full) > 40:
                self.df_obs_full = self.df_obs_full.iloc[20:-20].copy()

            mask_valid = (self.df_obs_full['Flux_mJy'] > 0) & (~np.isnan(self.df_obs_full['Flux_mJy']))
            mask_valid &= (self.df_obs_full['Flux_mJy'] / self.df_obs_full['Flux_err'] >= 7)
            df_obs = self.df_obs_full[mask_valid].copy()

            if len(df_obs) == 0:
                raise ValueError("No valid observational data points after quality masking.")

            self.obs_wave = df_obs['Wavelength'].values
            self.obs_flux = df_obs['Flux_mJy'].values * 1000.0  # mJy -> uJy
            self.obs_err = df_obs['Flux_err'].values * 1000.0    # mJy -> uJy

            # Relative weighting mask
            weights_spec = np.zeros_like(self.obs_wave)
            weights_spec[(self.obs_wave > 1.6) & (self.obs_wave < 2.2)] = 100
            weights_spec[(self.obs_wave > 2.2) & (self.obs_wave < 2.65)] = 1000
            weights_spec[(self.obs_wave > 3.9) & (self.obs_wave < 4.2)] = 100
            weights_spec[(self.obs_wave > 4.35) & (self.obs_wave < 4.55)] = 100

            safe_err = np.where(self.obs_err <= 0, np.median(self.obs_err[self.obs_err > 0]), self.obs_err)
            self.fit_weights = weights_spec / (safe_err**2)
            if np.sum(self.fit_weights) == 0:
                self.fit_weights = 1.0 / (safe_err**2)

            self.extinction_curve_fit = load_extinction_law(extinction_csv, self.obs_wave)
            self.temps, self.flux_on_fit, self.filenames = load_models(model_folder, self.obs_wave)

            # Update sliders
            min_t, max_t = float(self.temps.min()), float(self.temps.max())
            self.teff_slider.config(from_=min_t, to=max_t)
            self.teff_val.set((min_t + max_t) / 2)

            # Pre-estimate Scale
            init_mod = apply_extinction(self._get_interpolated_model(self.teff_val.get()), self.extinction_curve_fit, self.ak_val.get())
            denom = np.sum(self.fit_weights * (init_mod**2))
            opt_scale = np.sum(self.fit_weights * self.obs_flux * init_mod) / denom if denom > 0 else 1e-5
            opt_log_scale = np.log(max(opt_scale, 1e-30))

            self.scale_slider.config(from_=opt_log_scale - 15.0, to=opt_log_scale + 15.0)
            self.log_scale_val.set(opt_log_scale)

            self.status_label.config(text="Status: Loaded Successfully", foreground="green")
            self.on_slider_move()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text="Status: Load Failed", foreground="red")

    def _get_interpolated_model(self, teff):
        idx = np.searchsorted(self.temps, teff)
        if idx == 0:
            return self.flux_on_fit[0]
        elif idx == len(self.temps):
            return self.flux_on_fit[-1]
        else:
            t0, t1 = self.temps[idx-1], self.temps[idx]
            f0, f1 = self.flux_on_fit[idx-1], self.flux_on_fit[idx]
            return f0 + (f1 - f0) * ((teff - t0) / (t1 - t0))

    def on_slider_move(self, _=None):
        self.teff_label.config(text=f"{int(self.teff_val.get())} K")
        self.ak_label.config(text=f"{self.ak_val.get():.2f} mag")
        
        log_s = self.log_scale_val.get()
        self.scale_label.config(text=f"{log_s:.2f} (scale: {np.exp(log_s):.2e})")

        if self.obs_wave is not None:
            self.update_plot()

    def update_plot(self):
        teff = self.teff_val.get()
        ak = self.ak_val.get()
        scale = np.exp(self.log_scale_val.get())

        base_mod = self._get_interpolated_model(teff)
        flux_ext = apply_extinction(base_mod, self.extinction_curve_fit, ak)
        mod_scaled = flux_ext * scale

        self.ax.clear()
        wave_plot, flux_plot = break_at_gaps(self.obs_wave, self.obs_flux)
        
        self.ax.plot(wave_plot, flux_plot, label="Observed Spectrum", color="#55A868", lw=1.5)
        self.ax.plot(self.obs_wave, mod_scaled, label=f"Model (T={int(teff)}K, Ak={ak:.1f})", color="#C44E52", lw=1.5)

        self.ax.set_xlabel(r"Wavelength ($\mu\mathrm{m}$)")
        self.ax.set_ylabel(r"Flux Density ($\mu\mathrm{Jy}$)")
        self.ax.legend(loc="lower center", ncol=2)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    # ----------------- Threaded MCMC Pipeline -----------------

    def start_mcmc_thread(self):
        if self.obs_wave is None:
            messagebox.showwarning("Warning", "Please load spectrum data first.")
            return

        self.btn_mcmc.config(state="disabled")
        self.status_label.config(text="Status: Running MCMC...", foreground="orange")
        threading.Thread(target=self._run_mcmc_pipeline, daemon=True).start()

    def _run_mcmc_pipeline(self):
        try:
            obs_csv = self.obs_file_path.get()
            model_folder = self.model_folder_path.get()
            extinction_csv = self.extinction_path.get()
            fits_folder = self.output_folder_path.get()
            os.makedirs(fits_folder, exist_ok=True)

            base_filename = os.path.basename(obs_csv)
            id_match = re.search(r"(\d+)_obs", base_filename)
            source_name = id_match.group(1) if id_match else "Unknown"

            min_temp, max_temp = self.temps.min(), self.temps.max()

            def log_prior(theta):
                teff, ak, log_scale = theta
                if min_temp <= teff <= max_temp and 0.0 <= ak <= 30.0 and -100.0 < log_scale < 100.0:
                    return 0.0
                return -np.inf

            def log_likelihood(theta):
                teff, ak, log_scale = theta
                scale = np.exp(log_scale)
                base_mod = self._get_interpolated_model(teff)
                flux_ext = apply_extinction(base_mod, self.extinction_curve_fit, ak)
                mod_scaled = flux_ext * scale
                
                diff = self.obs_flux - mod_scaled
                chi2 = np.sum(self.fit_weights * (diff**2))
                return -0.5 * chi2

            def log_probability(theta):
                lp = log_prior(theta)
                return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

            # MLE Optimization
            init_teff = self.teff_val.get()
            init_ak = self.ak_val.get()
            init_log_scale = self.log_scale_val.get()

            nll = lambda theta: -log_likelihood(theta) if np.isfinite(log_prior(theta)) else 1e12
            res = minimize(nll, [init_teff, init_ak, init_log_scale], method='Nelder-Mead')
            mle_teff, mle_ak, mle_log_scale = res.x
            mle_ak = np.clip(mle_ak, 0.05, 29.5)

            # Walkers Initialization
            ndim, nwalkers = 3, 32
            pos = []
            for _ in range(nwalkers):
                p_teff = np.clip(mle_teff + np.random.normal(0, 10.0), min_temp + 1, max_temp - 1)
                p_ak = np.clip(mle_ak + np.random.normal(0, 0.05), 0.01, 29.9)
                p_lscale = mle_log_scale + np.random.normal(0, 0.02)
                pos.append(np.array([p_teff, p_ak, p_lscale]))

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
            sampler.run_mcmc(pos, 2000, progress=False)

            flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)
            teff_mcmc, ak_mcmc, log_scale_mcmc = np.percentile(flat_samples, 50, axis=0)
            teff_err, ak_err, log_scale_err = np.std(flat_samples, axis=0)
            best_scale = np.exp(log_scale_mcmc)

            teff_err = np.sqrt((teff_err**2) + (100**2))
            ak_err = np.sqrt((ak_err**2) + (0.15**2))

            best_idx = np.argmin(np.abs(self.temps - teff_mcmc))
            best_fn = self.filenames[best_idx]

            # Save Fits & Generate Publication Figures
            wave_full = self.df_obs_full['Wavelength'].values
            flux_full = self.df_obs_full['Flux_mJy'].values * 1000.0
            err_full = self.df_obs_full['Flux_err'].values * 1000.0

            mod_full = pd.read_csv(os.path.join(model_folder, best_fn))
            mod_wave_full, mod_flux_full = mod_full['wavelength_micron'].values, mod_full['flux'].values
            mod_flux_full_uJy = mod_flux_full * (mod_wave_full ** 2) * FLUX_CONV_FACTOR

            ext_curve_full = load_extinction_law(extinction_csv, wave_full)
            flux_ext_full = apply_extinction(np.interp(wave_full, mod_wave_full, mod_flux_full_uJy), ext_curve_full, ak_mcmc)
            model_scaled_full = flux_ext_full * best_scale

            df_fit = pd.DataFrame({
                'wave_obs (microns)': wave_full,
                'flux_obs (muJy)': flux_full,
                'Continumn (muJy)': model_scaled_full,
                'flux_err': err_full
            })
            df_fit.to_csv(os.path.join(fits_folder, base_filename), index=False)

            # Plot Corner if Available
            if corner is not None and np.all(np.ptp(flat_samples, axis=0) > 0):
                labels = [r"$T_{\mathrm{eff}}$ (K)", r"$A_K$ (mag)", r"$\ln(\mathrm{scale})$"]
                fig_corner = corner.corner(
                    flat_samples, labels=labels, truths=[teff_mcmc, ak_mcmc, log_scale_mcmc],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True
                )
                fig_corner.savefig(os.path.join(fits_folder, base_filename.replace('.csv', '_corner.png')), bbox_inches='tight', dpi=300)
                plt.close(fig_corner)

            # Linear and Log Fits Plots
            fig_out, ax1 = plt.subplots(figsize=(12, 5))
            wave_flux, flux_plot = break_at_gaps(wave_full, flux_full, gap_factor=5)
            teff_display = int(round(teff_mcmc / 50.0) * 50)

            label_mod = (
                r"Best-fit Model" "\n"
                rf"  $T_{{\mathrm{{eff}}}} = {teff_display} \pm {teff_err:.0f}\mathrm{{\ K}}$" "\n"
                rf"  $A_K = {ak_mcmc:.1f} \pm {ak_err:.1f}\mathrm{{\ mag}}$"
            )

            mask_ext = (mod_wave_full >= 1.35) & (mod_wave_full <= 5.1)
            w_highres = mod_wave_full[mask_ext]
            f_highres_uJy = mod_flux_full_uJy[mask_ext]
            f_highres = apply_extinction(f_highres_uJy, load_extinction_law(extinction_csv, w_highres), ak_mcmc) * best_scale

            ax1.plot(w_highres, f_highres, c='#C44E52', lw=1.5, alpha=0.9, label=label_mod, zorder=4)
            ax1.plot(wave_flux, flux_plot, c='#55A868', lw=2, alpha=0.85, label='Observed Spectrum', zorder=3)
            ax1.set_xlabel(r"Wavelength ($\mu\mathrm{m}$)")
            ax1.set_ylabel(r"Flux Density ($\mu\mathrm{Jy}$)")
            ax1.set_xlim(1.35, 5.1)
            ax1.legend(loc='lower center', ncol=2)

            out_png = os.path.join(fits_folder, base_filename.replace('.csv', '_fit.png'))
            fig_out.savefig(out_png, bbox_inches='tight', dpi=300)
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=0.1)
            fig_out.savefig(out_png.replace('.png', '_log.png'), bbox_inches='tight', dpi=300)
            plt.close(fig_out)

            # Update GUI Elements
            self.root.after(0, self._finish_mcmc, teff_mcmc, ak_mcmc, log_scale_mcmc)

        except Exception as e:
            self.root.after(0, self._fail_mcmc, str(e))

    def _finish_mcmc(self, teff, ak, log_scale):
        self.teff_val.set(teff)
        self.ak_val.set(ak)
        self.log_scale_val.set(log_scale)
        self.on_slider_move()
        self.btn_mcmc.config(state="normal")
        self.status_label.config(text="Status: MCMC Complete!", foreground="green")
        messagebox.showinfo("Success", f"MCMC fitting completed successfully.\nResults saved to '{self.output_folder_path.get()}'.")

    def _fail_mcmc(self, err_msg):
        self.btn_mcmc.config(state="normal")
        self.status_label.config(text="Status: Fit Failed", foreground="red")
        messagebox.showerror("MCMC Error", err_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrumFitterGUI(root)
    root.mainloop()
