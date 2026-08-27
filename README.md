# bilby_xG

**Gravitational-wave inference with next-generation detectors, built on
[bilby](https://git.ligo.org/lscsoft/bilby).**

`bilby_xG` extends bilby to the regime of next-generation ground-based
gravitational-wave detectors — Cosmic Explorer and the Einstein Telescope —
where the assumptions built into standard parameter estimation break down.
Signals in these detectors last long enough that the Earth rotates
appreciably while they are in band, and the detectors are large enough that
the long-wavelength approximation fails at high frequencies. `bilby_xG`
provides the frequency-dependent, finite-size, Earth-rotation-aware antenna
response (Baral et al. 2023, arXiv:2304.09889), matching likelihoods
(standard, multi-banded and relative-binning, with higher-order-mode
support), individual-mode source models, and the Cosmic Explorer detector
definitions and sensitivity curves needed to use them out of the box.

Beyond general relativity, it also supports two propagation tests through
the same likelihoods: a constant speed of gravity `vG` and a
Lorentz-violating modified dispersion relation `(a, A)`, selected
automatically from the parameters you sample.

The package is a stand-alone layer over an unmodified `bilby` (plus
`bilby.cython`), extending it purely by subclassing: importing `bilby_xG`
never changes the behaviour of an existing bilby analysis.

Historically, these capabilities lived on three separate, mutually-incompatible
branches of a bilby fork (the antenna-response/higher-order-mode branch, the
`SoG` speed-of-gravity branch, and the `dispersion` branch); `bilby_xG`
unifies them behind a single
[`Propagation`](bilby_xG/propagation.py) interface.

## Installation

```bash
pip install bilby bilby.cython   # if not already present
pip install -e .                 # from this repository
```

## Structure

The package mirrors bilby's module layout:

| Module | Contents |
|---|---|
| `bilby_xG.geometry` | `InterferometerGeometry` with the per-arm detector tensors `xx`/`yy` |
| `bilby_xG.interferometer` | `Interferometer` with the frequency-dependent, finite-size, Earth-rotation-aware response and a `vG`-aware `time_delay_from_geocenter` |
| `bilby_xG.networks` | `get_empty_interferometer`, `InterferometerList`, `TriangularInterferometer`, `PowerSpectralDensity` resolving the shipped `CE`/`CE20`/`ET` definitions and noise curves |
| `bilby_xG.likelihood` | `GravitationalWaveTransientNextGeneration` and its multi-banded and relative-binning variants |
| `bilby_xG.source` | CBC source models, including the individual-mode (higher-order-mode) models |
| `bilby_xG.propagation` | `Propagation` (GR), `SpeedOfGravity`, `ModifiedDispersion`, `build_propagation` |
| `bilby_xG.conversion` | CBC parameter conversions that also accept `log10_luminosity_distance` |
| `bilby_xG.utils` | time-to-merger estimate and shipped-data lookup helpers |

`bilby` itself is never modified: everything is provided through subclasses, so
existing bilby analyses are completely unaffected by importing `bilby_xG`.

## Usage

Build the detectors through `bilby_xG.networks` (rather than
`bilby.gw.detector`) so they carry the frequency-dependent response; everything
else is standard bilby:

```python
import bilby
from bilby_xG.networks import InterferometerList
from bilby_xG.likelihood import GravitationalWaveTransientNextGeneration
from bilby_xG.source import lal_binary_black_hole_individual_modes

# Next-generation detectors resolve by name; bilby's built-in names
# (H1, L1, V1, ...) also work and return bilby_xG interferometers.
ifos = InterferometerList(["CE", "CE20"])

likelihood = GravitationalWaveTransientNextGeneration(
    interferometers=ifos, waveform_generator=waveform_generator,
)
```

To sample in `log10_luminosity_distance`, pass the bilby_xG converter to the
waveform generator:

```python
import bilby_xG.conversion

waveform_generator = bilby.gw.WaveformGenerator(
    ...,
    parameter_conversion=bilby_xG.conversion.convert_to_lal_binary_black_hole_parameters,
)
```

### Choosing the physics

The propagation model is selected automatically from the sampled parameters:

| Sampled parameters | Model |
|---|---|
| neither `vG` nor `a, A` | general relativity (default) |
| `vG` | constant speed of gravity |
| `a` and `A` | modified dispersion relation |

You can also build a model directly:

```python
from bilby_xG.propagation import SpeedOfGravity, ModifiedDispersion
```

## Shipped data

`bilby_xG` ships next-generation detector definitions — Cosmic Explorer
(`CE`, `CE20`) and the latest Einstein Telescope configurations: the 10 km
triangle at the Euregio Meuse–Rhine site (`ET-EMR`) and the 15 km L-shaped
detectors in Sardinia (`ET_1L_IT`) and Lusatia (`ET_1L_DE`) — together with
their amplitude/power spectral densities (`CE`/`CE20` ASDs, `ET_10_HF_psd_pub.txt`,
`ET_15_HF_psd_pub.txt`), resolved automatically by `bilby_xG.networks`.

> **Note:** `bilby_xG` ships an updated 40 km `CE` definition that **takes
> precedence** over bilby's built-in `CE` when using `bilby_xG.networks`.
> Use `bilby.gw.detector.get_empty_interferometer("CE")` if you need the
> original.

## Status / caveats

- The relative-binning `*NextGeneration` likelihoods are large; validate
  against a general-relativity baseline for your configuration before
  production use.
- `MBGravitationalWaveTransientNextGeneration`'s `response_update` option is
  not implemented (only the default per-evaluation update is supported).
- The modified-dispersion propagation phase is implemented in its correct
  frequency-dependent form (a debugging `f = 100` override present on the
  original branch has been removed).

## License

MIT. See [LICENSE](LICENSE).
