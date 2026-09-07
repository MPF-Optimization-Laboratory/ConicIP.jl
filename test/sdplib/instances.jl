# SDPLIB instance catalogue and cache
# ===================================
#
# Shared by `test/sdplib_tests.jl` (the CI gate) and `benchmark/sdplib.jl`.
# Nothing is vendored: instances are fetched at run time from a pinned
# upstream commit and cached in a scratch space owned by ConicIP, so the
# same file is downloaded at most once per machine and shared between the
# test suite and the benchmark script.
#
# See test/sdplib/README.md for provenance, the citation, the reference
# objective values and the licence position.

# Both stdlibs, so this file is self-contained in either the test
# environment or the package environment.
using Downloads, SHA

# vsdp/SDPLIB commit the hashes below were taken from. Pinned, not `master`:
# the gate must not change meaning because upstream moved.
const SDPLIB_COMMIT = "fa11b45c1d8c896a6abad2648d5dad46d8ecefaa"

# Overridable so the offline/skip path can be exercised deliberately.
const SDPLIB_BASE_URL = get(ENV, "CONICIP_SDPLIB_URL",
    "https://raw.githubusercontent.com/vsdp/SDPLIB/$SDPLIB_COMMIT/data")

# ConicIP's package UUID, used to locate the scratch space without needing
# the module itself (benchmark/sdplib.jl does not load ConicIP in its driver
# process).
const SDPLIB_PKG_UUID = Base.UUID("d92ec50d-21ac-4ea5-850a-0588cd9e47b8")

# sha256 of each instance at SDPLIB_COMMIT. A file that does not match is
# deleted and the caller fails loudly: a wrong file must never be solved.
const SDPLIB_SHA256 = Dict(
    # the six used by the test gate
    "control1" => "482528bb128e64dad102fab88e4e8b7074efdfa22e396ebec586d832b1545bcb",
    "hinf1"    => "a2d3e9f340f304fe59147e5f7d8b3c54c8169cebe946d81009796c184164ab77",
    "infd1"    => "4cbb4dcd44caa57c6970db23905971ed144f1046b663dfb828decda51d12acd8",
    "infp1"    => "c81f23ce297cd489c0500076677d6c70727fb1e761ca21d53398498e8192dd45",
    "theta1"   => "e957517b2284f24eba158db56a0ae34ecc07d24fa299a31f732dad3d4a54ea34",
    "truss1"   => "07bfaa5beaee8d2df2188a7aff80abe307a176466824211d68ffe68764c6efca",
    # the rest of the benchmark list (benchmark/sdplib.jl)
    "arch0"    => "2e87189c77823fafa2755f4fd6d0a2dd6476f06297a2d0d9a017b95ade3943bd",
    "control2" => "3a43871daec9c0e1700caba254708c61379eff58be78576da4a2a32115274faa",
    "control3" => "931c6f1cb5b70ad7907d73067642c935228244c3908ebc4912001271d0fdb009",
    "gpp100"   => "e64caafc501b0c26917a80e167aa1c39b6e01ccddbd077acbd14971848ab1602",
    "hinf2"    => "16f42918cb14d9eecd6c7f881d6fd16f8eb760c9450182e32c894a75204b45ef",
    "hinf3"    => "714efcea520b6c70d43ea072302a704d2201bc93023b6b799f0efb84ed4a492f",
    "mcp100"   => "a33665823d81f4ba1285272b355cefc2d3307a1f5fb8bb933edee58b3615a9b8",
    "mcp124-1" => "7f93d4213b4f49ffb27c9f1bec4b0ac56b5b7c8c3662a5def5fd9ee8aa289ae7",
    "qap5"     => "08afd61ec131d190aa3344f3bfd5c39551b1a5639b99f42ecf5b6a993faa7a52",
    "qap6"     => "3e1d035f90184c62f275c817a1fc5920c4a8bad1beda344e401a239c79867974",
    "theta2"   => "3c837c54400e6e442cb73b6ca55be5f4eb8e46c8b1c27ed9c379d2c632b5cc73",
    "truss2"   => "a644092ee475e7843c89851769b08ac48c105888b8e56547967dad4a7e5a7c75",
    "truss3"   => "c021b04541999b406434aea8cab6b09bde307924e737a07c8d340823cc4a8ede",
    "truss4"   => "7b9c1e1b9c535308dcceaa0dcf9b06cee7e1ef03f3f44c089de7ba44036e0feb",
)

# The six instances the CI gate solves.
const SDPLIB_TEST_INSTANCES =
    ("truss1", "hinf1", "control1", "theta1", "infp1", "infd1")

# Directory the instances are cached in. `Scratch` is a test-only dependency,
# so fall back to the directory it would have created — the layout
# (<depot>/scratchspaces/<uuid>/<key>) is Scratch's documented contract — and
# both entry points then share one cache.
function sdplib_cache_dir()
    if Base.identify_package("Scratch") !== nothing
        Scratch = Base.require(Base.PkgId(
            Base.UUID("6c6a2e73-6563-6170-7368-637461726353"), "Scratch"))
        return Base.invokelatest(Scratch.get_scratch!, SDPLIB_PKG_UUID, "sdplib")
    end
    dir = joinpath(first(DEPOT_PATH), "scratchspaces",
                   string(SDPLIB_PKG_UUID), "sdplib")
    mkpath(dir)
    return dir
end

# Raised when a cached or freshly downloaded file does not match its
# recorded hash. A distinct type so callers can tell corrupt data (always
# fatal) from a network failure (which may legitimately be skipped).
struct SDPLIBHashMismatch <: Exception
    name::String
    expected::String
    got::String
    url::String
end

function Base.showerror(io::IO, e::SDPLIBHashMismatch)
    print(io, """
          SDPLIB instance $(e.name) failed its sha256 check and was deleted.
            expected $(e.expected)
            got      $(e.got)
          Source: $(e.url) (pinned commit $SDPLIB_COMMIT).
          Re-run to fetch it again; if it keeps failing, the upstream file or
          the recorded hash is wrong.""")
end

# Path to `name`, downloading it into `dir` if it is not already cached.
#
# Throws on a download failure (the caller decides whether that is a skip or
# a hard failure) and on a hash mismatch, after removing the bad file. A
# cached file whose hash no longer matches is treated the same way: deleted
# and reported, never used.
function sdplib_fetch!(name; dir = sdplib_cache_dir())
    want = get(SDPLIB_SHA256, name, nothing)
    want === nothing && error("no sha256 recorded for SDPLIB instance $name")
    path = joinpath(dir, "$name.dat-s")

    if !isfile(path)
        url = "$SDPLIB_BASE_URL/$name.dat-s"
        tmp = path * ".part"
        try
            Downloads.download(url, tmp)
        catch
            rm(tmp; force = true)
            rethrow()
        end
        mv(tmp, path; force = true)
    end

    got = open(SHA.sha256, path) |> bytes2hex
    if got != want
        rm(path; force = true)
        throw(SDPLIBHashMismatch(name, want, got, "$SDPLIB_BASE_URL/$name.dat-s"))
    end
    return path
end
