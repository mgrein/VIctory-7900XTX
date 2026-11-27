//! Series X SDR 1440p and 4K profiles for the title's FidelityFX upscaler,
//! run by the host's PC build of it (`victory_hle::modules::fidelityfx`).
//! Off by default; one of these switches turns it on:
//!
//! - `VICTORY_FSR_4K=<quality|balanced|performance|1080p|1>`: output
//!   3840x2160 (`1` is quality: 2560x1440 rendered).
//! - `VICTORY_FSR_1440P=<1080p|quality|balanced|performance|1>`: output
//!   2560x1440 (`1` is 1080p: 1920x1080 rendered).
//! - `VICTORY_FSR=<preset>` with `VICTORY_OUTPUT=2560x1440|3840x2160`
//!   (4K without it).
//!
//! `0` or empty is off. Each needs `VICTORY_FIDELITYFX_DIR`: the folder of
//! the user's own `amd_fidelityfx_upscaler_dx12.dll` (AMD's signed DirectX 12
//! upscaler from the FidelityFX SDK, `Kits/FidelityFX/signedbin`), or the
//! DLL's path. Series X only, and only for the one retail build whose code
//! the patches below were checked against (its hash): every patched place is
//! compared first, and anything unexpected stops the run with an error. The
//! patched image is cached apart (`images/fsr-<preset>-<output>-sdr`), and
//! `VICTORY_OUTPUT`'s own patching (`victory_loader::resolution`) is left
//! out. The bridge asks the SDK for FSR 4.1.2 and, where it offers none
//! (NVIDIA), for its newest FSR 3 or later (3.1); untested on NVIDIA.
//!
//! Ported from VIctory-DXR 443fd629 (Maren Halvorsen; the 1440p profiles)
//! and Belobug cd8dfc06 (beloooo), both after DexNom ab34a8b2.
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use victory_loader::PreparedImage;
use victory_loader::resolution::{self, OUTPUT, POOL_CHECK, output_code, pool_check_code};

const RETAIL_BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const SDR_MODE: u32 = 0x0615_c100;
const STOCK_SDR_MODE: [u8; 16] = [0, 15, 0, 0, 112, 8, 0, 0, 60, 0, 0, 0, 1, 0, 0, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FsrOutput {
    Qhd1440p,
    Uhd4k,
}

impl FsrOutput {
    pub(super) fn dimensions(self) -> [u32; 2] {
        match self {
            Self::Qhd1440p => [2560, 1440],
            Self::Uhd4k => [3840, 2160],
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Qhd1440p => "1440p",
            Self::Uhd4k => "4k",
        }
    }

    pub(super) fn display_mode(self) -> &'static str {
        match self {
            Self::Qhd1440p => "2560x1440",
            Self::Uhd4k => "3840x2160",
        }
    }

    pub(super) fn from_dimensions(size: (u32, u32)) -> Option<Self> {
        match size {
            (2560, 1440) => Some(Self::Qhd1440p),
            (3840, 2160) => Some(Self::Uhd4k),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FsrPreset {
    Render1080p,
    Quality,
    Balanced,
    Performance,
}

impl FsrPreset {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "1080p" => Some(Self::Render1080p),
            "quality" => Some(Self::Quality),
            "balanced" => Some(Self::Balanced),
            "performance" => Some(Self::Performance),
            _ => None,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Render1080p => "1080p",
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
        }
    }

    // AMD SDK FFX_UPSCALE_QUALITY_MODE values.
    pub(super) fn mode(self, output: FsrOutput) -> u8 {
        match (self, output) {
            (Self::Render1080p, FsrOutput::Qhd1440p) => 1,
            (Self::Render1080p, FsrOutput::Uhd4k) => 3,
            (Self::Quality, _) => 1,
            (Self::Balanced, _) => 2,
            (Self::Performance, _) => 3,
        }
    }

    pub(super) fn render_scale(self, output: FsrOutput) -> f32 {
        match (self, output) {
            (Self::Render1080p, FsrOutput::Qhd1440p) => 0.75,
            (Self::Render1080p, FsrOutput::Uhd4k) => 0.5,
            (Self::Quality, _) => 2.0 / 3.0,
            (Self::Balanced, _) => 1.0 / 1.7,
            (Self::Performance, _) => 0.5,
        }
    }

    pub(super) fn render_size(self, output: FsrOutput) -> [u32; 2] {
        let scale = self.render_scale(output);
        output.dimensions().map(|size| (size as f32 * scale) as u32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FsrProfile {
    pub(super) output: FsrOutput,
    pub(super) preset: FsrPreset,
}

impl FsrProfile {
    pub(super) fn requested() -> Result<Option<Self>> {
        Self::from_env(|key| match std::env::var(key) {
            Ok(v) => Ok(Some(v)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(err) => Err(err.into()),
        })
    }

    pub(super) fn from_env(
        mut env: impl FnMut(&str) -> Result<Option<String>>,
    ) -> Result<Option<Self>> {
        let active = |val: Option<String>| -> Option<String> {
            val.and_then(|raw| {
                let trimmed = raw.trim();
                (!trimmed.is_empty() && trimmed != "0").then(|| trimmed.to_owned())
            })
        };
        let fsr_1440p = active(env("VICTORY_FSR_1440P")?);
        let fsr_generic = active(env("VICTORY_FSR")?);
        let fsr_4k = active(env("VICTORY_FSR_4K")?);
        let output_env = env("VICTORY_OUTPUT")?
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());

        let count = usize::from(fsr_1440p.is_some())
            + usize::from(fsr_generic.is_some())
            + usize::from(fsr_4k.is_some());
        if count == 0 {
            return Ok(None);
        }
        ensure!(
            count == 1,
            "set only one of VICTORY_FSR_1440P, VICTORY_FSR, or VICTORY_FSR_4K"
        );

        let parse_fsr_output = |spec: &str| -> Result<FsrOutput> {
            resolution::parse_output(spec)
                .and_then(FsrOutput::from_dimensions)
                .with_context(|| {
                    format!("VICTORY_OUTPUT={spec}: FSR supports 2560x1440 or 3840x2160")
                })
        };

        if let Some(raw) = fsr_1440p {
            let lower = raw.to_ascii_lowercase();
            let preset = if lower == "1" {
                FsrPreset::Render1080p
            } else {
                FsrPreset::from_name(&lower).with_context(|| {
                    format!(
                        "VICTORY_FSR_1440P={raw}: expected 1, 1080p, quality, balanced, performance or 0"
                    )
                })?
            };
            if let Some(spec) = output_env {
                let parsed = parse_fsr_output(&spec)?;
                ensure!(
                    parsed == FsrOutput::Qhd1440p,
                    "VICTORY_FSR_1440P conflicts with VICTORY_OUTPUT={spec} (expected 2560x1440)"
                );
            }
            return Ok(Some(Self {
                output: FsrOutput::Qhd1440p,
                preset,
            }));
        }

        if let Some(raw) = fsr_4k {
            let lower = raw.to_ascii_lowercase();
            let preset = if lower == "1" {
                FsrPreset::Quality
            } else {
                FsrPreset::from_name(&lower).with_context(|| {
                    format!(
                        "VICTORY_FSR_4K={raw}: expected quality, balanced, performance, 1080p, 1 or 0"
                    )
                })?
            };
            if let Some(spec) = output_env {
                let parsed = parse_fsr_output(&spec)?;
                ensure!(
                    parsed == FsrOutput::Uhd4k,
                    "VICTORY_FSR_4K conflicts with VICTORY_OUTPUT={spec} (expected 3840x2160)"
                );
            }
            return Ok(Some(Self {
                output: FsrOutput::Uhd4k,
                preset,
            }));
        }

        let raw = fsr_generic.expect("exactly one FSR switch is active");
        let output = match output_env {
            Some(spec) => parse_fsr_output(&spec)?,
            None => FsrOutput::Uhd4k,
        };
        let lower = raw.to_ascii_lowercase();
        let preset = if lower == "1" {
            match output {
                FsrOutput::Qhd1440p => FsrPreset::Render1080p,
                FsrOutput::Uhd4k => FsrPreset::Quality,
            }
        } else {
            FsrPreset::from_name(&lower).with_context(|| {
                format!(
                    "VICTORY_FSR={raw}: expected quality, balanced, performance, 1080p, 1 or 0"
                )
            })?
        };
        Ok(Some(Self { output, preset }))
    }

    pub(super) fn mode(self) -> u8 {
        self.preset.mode(self.output)
    }

    pub(super) fn render_scale(self) -> f32 {
        self.preset.render_scale(self.output)
    }

    pub(super) fn render_size(self) -> [u32; 2] {
        self.preset.render_size(self.output)
    }

    pub(super) fn output_size(self) -> [u32; 2] {
        self.output.dimensions()
    }

    pub(super) fn cache_subdir(self) -> String {
        format!("images/fsr-{}-{}-sdr", self.preset.name(), self.output.label())
    }
}

pub(super) fn fidelityfx_dir() -> Result<PathBuf> {
    let val = std::env::var_os("VICTORY_FIDELITYFX_DIR");
    resolve_fidelityfx_dir(val.as_deref())
}

pub(super) fn resolve_fidelityfx_dir(env_value: Option<&OsStr>) -> Result<PathBuf> {
    let raw = env_value
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .context(
            "FSR 4.1.2 requires amd_fidelityfx_upscaler_dx12.dll. Set VICTORY_FIDELITYFX_DIR to its folder.",
        )?;
    normalize_fidelityfx_dir(&raw)
}

fn normalize_fidelityfx_dir(path: &Path) -> Result<PathBuf> {
    let dll = victory_hle::modules::fidelityfx::DLL;
    if path
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case(dll))
    {
        if !path.is_file() {
            bail!(
                "VICTORY_FIDELITYFX_DIR points to {}, which does not exist",
                path.display()
            );
        }
        return path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .context("VICTORY_FIDELITYFX_DIR DLL path has no parent directory");
    }
    let dll_path = path.join(dll);
    if !dll_path.is_file() {
        bail!(
            "VICTORY_FIDELITYFX_DIR={} does not contain {dll}",
            path.display()
        );
    }
    Ok(path.to_path_buf())
}

pub(super) fn prepare_fsr(image: &mut PreparedImage, profile: FsrProfile) -> Result<()> {
    ensure!(
        image.original_hash.to_hex().as_str() == RETAIL_BUILD,
        "FSR is unavailable for this retail game build"
    );
    ensure!(
        *field::<16>(image, SDR_MODE)? == STOCK_SDR_MODE,
        "Unexpected Series X SDR mode"
    );
    ensure!(
        *field::<4>(image, 0x0615_d95c)? == (2.0f32 / 3.0).to_le_bytes(),
        "Unexpected Series X FSR scene scale"
    );
    ensure!(
        *field::<8>(image, 0x007e_9421)? == [0xb8, 6, 0, 0, 0, 0x0f, 0x45, 0xc1],
        "Unexpected Series X temporal renderer selection"
    );
    ensure!(
        *field::<14>(image, 0x028e_f69a)?
            == [
                0x0f, 0xb6, 0xc0, 0x41, 0x89, 0x86, 0x90, 0, 0, 0, 0x84, 0xc0, 0x75, 0x0a
            ],
        "Unexpected Series X FSR render size selection"
    );
    const NATIVE_LOAD: u32 = 0x028e_f6a8;
    const QUALITY_LOAD: u32 = 0x028e_f6b2;
    let custom_scale = profile.preset != FsrPreset::Quality;
    if custom_scale {
        ensure!(
            *field::<10>(image, NATIVE_LOAD)?
                == [0xc5, 0xfa, 0x10, 0x05, 0x28, 0x78, 0x6d, 0x03, 0xeb, 0x08]
                && *field::<8>(image, QUALITY_LOAD)?
                    == [0xc5, 0xfa, 0x10, 0x05, 0xa2, 0xe2, 0x86, 0x03],
            "Unexpected Series X FSR scale instructions"
        );
    }
    let custom_output = profile.output != FsrOutput::Uhd4k;
    if custom_output {
        ensure!(
            field_slice(image, OUTPUT.at as u32, OUTPUT.bytes.len())? == OUTPUT.bytes,
            "Unexpected Series X output resolution site"
        );
        ensure!(
            field_slice(image, POOL_CHECK.at as u32, POOL_CHECK.bytes.len())? == POOL_CHECK.bytes,
            "Unexpected Series X back buffer pool check site"
        );
    }
    // The original selector has already run its capability check. An
    // explicit FSR session uses renderer 7 every time it queries this.
    *field::<8>(image, 0x007e_9421)? = [0xb8, 7, 0, 0, 0, 0x90, 0x90, 0x90];
    // Record the SDK mode and jump to the scaled-scene branch.
    *field::<14>(image, 0x028e_f69a)? = [
        0xb8,
        profile.mode(),
        0,
        0,
        0,
        0x41,
        0x89,
        0x86,
        0x90,
        0,
        0,
        0,
        0xeb,
        0x0a,
    ];
    if custom_scale {
        // The jump makes the native-scale instruction unreachable. Reuse
        // its RIP operand bytes for a private literal, leaving the constants
        // used by other passes unchanged.
        let local_scale = NATIVE_LOAD + 4;
        *field::<4>(image, local_scale)? = profile.render_scale().to_le_bytes();
        let displacement = local_scale as i32 - (QUALITY_LOAD as i32 + 8);
        *field::<4>(image, QUALITY_LOAD + 4)? = displacement.to_le_bytes();
    }
    if custom_output {
        let [w, h] = profile.output_size();
        let out_patch = output_code(w, h);
        field_slice(image, OUTPUT.at as u32, OUTPUT.bytes.len())?.copy_from_slice(&out_patch);
        let pool_patch = pool_check_code();
        field_slice(image, POOL_CHECK.at as u32, POOL_CHECK.bytes.len())?[..pool_patch.len()]
            .copy_from_slice(&pool_patch);
    }
    Ok(())
}

fn field<const N: usize>(image: &mut PreparedImage, rva: u32) -> Result<&mut [u8; N]> {
    field_slice(image, rva, N)?
        .try_into()
        .context("Invalid display profile field length")
}

fn field_slice(image: &mut PreparedImage, rva: u32, len: usize) -> Result<&mut [u8]> {
    let offset = image
        .info
        .rva_to_file_offset(rva)
        .context("Display profile field missing")? as usize;
    image
        .bytes
        .get_mut(offset..offset + len)
        .context("Incomplete display profile field")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ffi::OsString;

    #[test]
    fn fsr_preset_resolution_calculations() {
        let cases = [
            // 1440p presets (primary: 1920x1080 -> 2560x1440 at 0.75x)
            (
                FsrOutput::Qhd1440p,
                FsrPreset::Render1080p,
                [1920, 1080],
                [2560, 1440],
                1,
                "images/fsr-1080p-1440p-sdr",
            ),
            (
                FsrOutput::Qhd1440p,
                FsrPreset::Quality,
                [1706, 960],
                [2560, 1440],
                1,
                "images/fsr-quality-1440p-sdr",
            ),
            (
                FsrOutput::Qhd1440p,
                FsrPreset::Balanced,
                [1505, 847],
                [2560, 1440],
                2,
                "images/fsr-balanced-1440p-sdr",
            ),
            (
                FsrOutput::Qhd1440p,
                FsrPreset::Performance,
                [1280, 720],
                [2560, 1440],
                3,
                "images/fsr-performance-1440p-sdr",
            ),
            // 4K presets
            (
                FsrOutput::Uhd4k,
                FsrPreset::Quality,
                [2560, 1440],
                [3840, 2160],
                1,
                "images/fsr-quality-4k-sdr",
            ),
            (
                FsrOutput::Uhd4k,
                FsrPreset::Balanced,
                [2258, 1270],
                [3840, 2160],
                2,
                "images/fsr-balanced-4k-sdr",
            ),
            (
                FsrOutput::Uhd4k,
                FsrPreset::Performance,
                [1920, 1080],
                [3840, 2160],
                3,
                "images/fsr-performance-4k-sdr",
            ),
            (
                FsrOutput::Uhd4k,
                FsrPreset::Render1080p,
                [1920, 1080],
                [3840, 2160],
                3,
                "images/fsr-1080p-4k-sdr",
            ),
        ];
        for (output, preset, want_render, want_output, want_mode, want_cache) in cases {
            let profile = FsrProfile { output, preset };
            assert_eq!(profile.render_size(), want_render);
            assert_eq!(profile.output_size(), want_output);
            assert_eq!(profile.mode(), want_mode);
            assert_eq!(profile.cache_subdir(), want_cache);
        }
        assert_eq!(FsrPreset::Render1080p.render_scale(FsrOutput::Qhd1440p), 0.75);
        assert_eq!(FsrOutput::Qhd1440p.display_mode(), "2560x1440");
        assert_eq!(FsrOutput::Uhd4k.display_mode(), "3840x2160");
    }

    #[test]
    fn fsr_env_var_profile_selection() {
        let parse = |pairs: &[(&str, &str)]| {
            let map: HashMap<&str, String> = pairs
                .iter()
                .map(|&(k, v)| (k, v.to_owned()))
                .collect();
            FsrProfile::from_env(|k| Ok(map.get(k).cloned()))
        };

        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(parse(&[("VICTORY_FSR_1440P", "0")]).unwrap(), None);
        assert_eq!(parse(&[("VICTORY_FSR_4K", "0")]).unwrap(), None);
        assert_eq!(parse(&[("VICTORY_FSR", "0")]).unwrap(), None);

        // Primary 1440p switch: 1920x1080 -> 2560x1440
        let p = parse(&[("VICTORY_FSR_1440P", "1")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [1920, 1080]);
        assert_eq!(p.output_size(), [2560, 1440]);

        // 1440p Quality preset via VICTORY_FSR_1440P or VICTORY_FSR + VICTORY_OUTPUT
        let p = parse(&[("VICTORY_FSR_1440P", "quality")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [1706, 960]);
        assert_eq!(p.output_size(), [2560, 1440]);

        let p = parse(&[("VICTORY_FSR", "1080p"), ("VICTORY_OUTPUT", "2560x1440")])
            .unwrap()
            .unwrap();
        assert_eq!(p.render_size(), [1920, 1080]);
        assert_eq!(p.output_size(), [2560, 1440]);

        let p = parse(&[("VICTORY_FSR", "quality"), ("VICTORY_OUTPUT", "2560x1440")])
            .unwrap()
            .unwrap();
        assert_eq!(p.render_size(), [1706, 960]);
        assert_eq!(p.output_size(), [2560, 1440]);

        let p = parse(&[("VICTORY_FSR", "balanced"), ("VICTORY_OUTPUT", "2560x1440")])
            .unwrap()
            .unwrap();
        assert_eq!(p.render_size(), [1505, 847]);
        assert_eq!(p.output_size(), [2560, 1440]);

        let p = parse(&[
            ("VICTORY_FSR", "performance"),
            ("VICTORY_OUTPUT", "2560x1440"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(p.render_size(), [1280, 720]);
        assert_eq!(p.output_size(), [2560, 1440]);

        // 4K presets
        let p = parse(&[("VICTORY_FSR_4K", "quality")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [2560, 1440]);
        assert_eq!(p.output_size(), [3840, 2160]);

        let p = parse(&[("VICTORY_FSR_4K", "balanced")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [2258, 1270]);
        assert_eq!(p.output_size(), [3840, 2160]);

        let p = parse(&[("VICTORY_FSR_4K", "performance")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [1920, 1080]);
        assert_eq!(p.output_size(), [3840, 2160]);

        let p = parse(&[("VICTORY_FSR", "quality")]).unwrap().unwrap();
        assert_eq!(p.render_size(), [2560, 1440]);
        assert_eq!(p.output_size(), [3840, 2160]);

        // Matching VICTORY_OUTPUT alongside VICTORY_FSR_1440P is accepted.
        let p = parse(&[
            ("VICTORY_FSR_1440P", "1"),
            ("VICTORY_OUTPUT", "2560x1440"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(p.render_size(), [1920, 1080]);
        assert_eq!(p.output_size(), [2560, 1440]);

        // Conflicts and invalid values fail clearly.
        assert!(parse(&[("VICTORY_FSR_1440P", "1"), ("VICTORY_FSR_4K", "quality")]).is_err());
        assert!(parse(&[("VICTORY_FSR_1440P", "1"), ("VICTORY_OUTPUT", "3840x2160")]).is_err());
        assert!(parse(&[("VICTORY_FSR_4K", "quality"), ("VICTORY_OUTPUT", "2560x1440")]).is_err());
        assert!(parse(&[("VICTORY_FSR", "quality"), ("VICTORY_OUTPUT", "1920x1080")]).is_err());
        assert!(parse(&[("VICTORY_FSR", "ultra")]).is_err());
    }

    #[test]
    fn fsr_dll_path_resolution() {
        let missing = resolve_fidelityfx_dir(None).unwrap_err().to_string();
        assert!(missing.contains("VICTORY_FIDELITYFX_DIR"));
        assert!(resolve_fidelityfx_dir(Some(OsStr::new(""))).is_err());

        let root = std::env::temp_dir().join(format!(
            "victory-fsr-dll-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // Folder without the DLL is rejected.
        let err = resolve_fidelityfx_dir(Some(root.as_os_str()))
            .unwrap_err()
            .to_string();
        assert!(err.contains(victory_hle::modules::fidelityfx::DLL));

        // Folder containing the DLL resolves to that folder.
        let dll_path = root.join(victory_hle::modules::fidelityfx::DLL);
        std::fs::write(&dll_path, b"stub").unwrap();
        assert_eq!(
            resolve_fidelityfx_dir(Some(root.as_os_str())).unwrap(),
            root
        );

        // Direct path to the DLL normalizes to its parent folder.
        assert_eq!(
            resolve_fidelityfx_dir(Some(dll_path.as_os_str())).unwrap(),
            root
        );

        // Missing direct DLL path fails.
        let missing_dll = OsString::from(
            root.join("missing")
                .join(victory_hle::modules::fidelityfx::DLL),
        );
        assert!(resolve_fidelityfx_dir(Some(missing_dll.as_os_str())).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fsr_presets_leave_the_stock_scanout_and_settings_viewport() {
        let Some(path) = std::env::var_os("VICTORY_DISPLAY_PROFILE_TEST_IMAGE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        for (output, preset, size) in [
            (FsrOutput::Qhd1440p, FsrPreset::Render1080p, [1920, 1080]),
            (FsrOutput::Qhd1440p, FsrPreset::Quality, [1706, 960]),
            (FsrOutput::Uhd4k, FsrPreset::Quality, [2560, 1440]),
            (FsrOutput::Uhd4k, FsrPreset::Balanced, [2258, 1270]),
            (FsrOutput::Uhd4k, FsrPreset::Performance, [1920, 1080]),
        ] {
            let profile = FsrProfile { output, preset };
            let mut image = victory_loader::prepare(&bytes).unwrap();
            let scanout = *field::<5>(&mut image, 0x0247_ad4a).unwrap();
            let settings = *field::<8>(&mut image, 0x0580_48d4).unwrap();
            prepare_fsr(&mut image, profile).unwrap();
            assert_eq!(*field::<16>(&mut image, SDR_MODE).unwrap(), STOCK_SDR_MODE);
            assert_eq!(*field::<5>(&mut image, 0x0247_ad4a).unwrap(), scanout);
            assert_eq!(*field::<8>(&mut image, 0x0580_48d4).unwrap(), settings);
            assert_eq!(
                *field::<4>(&mut image, 0x007e_9422).unwrap(),
                7u32.to_le_bytes()
            );
            assert_eq!(
                *field::<4>(&mut image, 0x028e_f69b).unwrap(),
                u32::from(profile.mode()).to_le_bytes()
            );
            assert_eq!(profile.render_size(), size);
            let disp = i32::from_le_bytes(*field::<4>(&mut image, 0x028e_f6b6).unwrap());
            let reference = 0x028e_f6ba_u32.wrapping_add_signed(disp);
            let scale = f32::from_le_bytes(*field::<4>(&mut image, reference).unwrap());
            let [ow, oh] = profile.output_size();
            assert_eq!([ow as f32, oh as f32].map(|w| (w * scale) as u32), size);
            if output == FsrOutput::Qhd1440p {
                assert_eq!(
                    field_slice(&mut image, OUTPUT.at as u32, OUTPUT.bytes.len()).unwrap(),
                    output_code(2560, 1440)
                );
                assert_eq!(
                    &field_slice(&mut image, POOL_CHECK.at as u32, POOL_CHECK.bytes.len()).unwrap()
                        [..8],
                    pool_check_code()
                );
            }
        }
    }
}
