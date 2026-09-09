# ──────────────────────────────────────────────────────────────
#  Gondzio multiple centrality correctors: spectral primitives
#
#  The corrector loop in _conicIP works on the trial Nesterov-Todd
#  complementarity w = ṽ ∘ s̃ and needs two maps that act on the
#  Jordan-algebraic eigenvalues of w, in the Jordan frame of w:
#
#    clip_spectral!           λᵢ ↦ clamp(λᵢ, lo, hi)          (Π_box(w))
#    centrality_correction!   λᵢ ↦ max(clamp(λᵢ, lo, hi) − λᵢ, −cap)
#
#  The second is Gondzio's corrector target Π_box(w) − w with the large
#  negative components capped at −cap (cap = βmax·σμ). Because Π_box(w)
#  shares the frame of w, the difference has eigenvalues clamp(λᵢ) − λᵢ in
#  that frame and the cap is a clip of those from below — one pass.
#
#  Frames per cone (spectral_map! below):
#    "R"  each entry is its own eigenvalue.
#    "Q"  λ± = w₁ ± ‖w̄‖ with idempotents c± = ½(1, ±w̄/‖w̄‖); when
#         ‖w̄‖ = 0 the frame is taken along e₂ (any unit vector works and
#         the two eigenvalues then coincide, so the reassembled w̄ is 0).
#    "S"  eigen(Symmetric(mat(w))), reassemble V·diag(f(λ))·Vᵀ, vecm back.
#  A block whose eigenvalues f leaves unchanged is copied verbatim, so the
#  identity map is exact (no mat/vecm round trip) and cheap.
# ──────────────────────────────────────────────────────────────

# Gondzio corrector parameters (Gondzio 1996; Colombo & Gondzio 2008).
const GONDZIO_δα   = 0.1    # step-length aspiration α̃ = min(1, α + δα)
const GONDZIO_βmin = 0.1    # target box [βmin·σμ, βmax·σμ] for the
const GONDZIO_βmax = 10.0   #   trial complementarity
const GONDZIO_γ    = 0.1    # accept when α_new ≥ α + γ(α̃ − α)

"""
    spectral_map!(out, w, f, cone_dims)

Apply the scalar map `f` to the Jordan-algebraic eigenvalues of `w`, block
by block over the cone product `cone_dims`, keeping the Jordan frame of
`w`. Writes the result into `out` (which may alias `w`) and returns it.
"""
function spectral_map!(out, w, f, cone_dims)
  off = 0
  @inbounds for (btype, k) in cone_dims
    I = off+1:off+k
    wI = view(w, I); oI = view(out, I)
    if btype == "R"
      for j in eachindex(wI); oI[j] = f(wI[j]); end
    elseif btype == "Q"
      _spectral_map_soc!(oI, wI, f)
    elseif btype == "S"
      _spectral_map_sdc!(oI, wI, f)
    else
      throw(ArgumentError("unknown cone type $(btype)"))
    end
    off += k
  end
  return out
end

function _spectral_map_soc!(o, w, f)
  k = length(w)
  w1 = w[1]
  nb = 0.0
  @inbounds for j in 2:k; nb += w[j]*w[j]; end
  nb = sqrt(nb)
  λp = w1 + nb; λm = w1 - nb
  fp = f(λp);   fm = f(λm)
  if fp == λp && fm == λm
    o === w || copyto!(o, w)
    return o
  end
  o[1] = (fp + fm)/2
  d = (fp - fm)/2
  if nb > 0
    @inbounds for j in 2:k; o[j] = d*(w[j]/nb); end
  else
    # Degenerate frame along e₂: λ₊ = λ₋ here, so d = 0 and w̄ stays 0;
    # writing the direction explicitly keeps the formula total.
    @inbounds for j in 2:k; o[j] = 0.0; end
    k >= 2 && (o[2] = d)
  end
  return o
end

function _spectral_map_sdc!(o, w, f)
  isempty(w) && return o
  E = eigen(Symmetric(mat(w)))
  Λ = E.values
  changed = false
  @inbounds for j in eachindex(Λ)
    fj = f(Λ[j])
    if fj != Λ[j]; Λ[j] = fj; changed = true; end
  end
  if !changed
    o === w || copyto!(o, w)
    return o
  end
  V = E.vectors
  Z = (V .* Λ') * V'
  vecm!(o, (Z .+ Z') ./ 2)
  return o
end

"""
    clip_spectral!(out, w, lo, hi, cone_dims)

Project the Jordan-algebraic eigenvalues of `w` onto `[lo, hi]` in the
frame of `w`, block by block over `cone_dims` (`"R"`: entrywise clamp;
`"Q"`: both eigenvalues `w₁ ± ‖w̄‖`; `"S"`: the eigenvalues of `mat(w)`).
The result is written into `out` and returned; `out` may alias `w`.
"""
function clip_spectral!(out, w, lo, hi, cone_dims)
  lo <= hi || throw(ArgumentError("clip_spectral!: lo = $lo exceeds hi = $hi"))
  return spectral_map!(out, w, λ -> clamp(λ, lo, hi), cone_dims)
end

"""
    centrality_correction!(Δw, w, lo, hi, cap, cone_dims)

Gondzio's corrector target for the trial complementarity `w`: the
spectral difference `Π_[lo,hi](w) − w`, with every component below `−cap`
raised to `−cap` (in the frame of `w`). Returns `Δw`; the correction is
identically zero when every eigenvalue of `w` already lies in the box.
"""
function centrality_correction!(Δw, w, lo, hi, cap, cone_dims)
  lo <= hi || throw(ArgumentError("centrality_correction!: lo = $lo exceeds hi = $hi"))
  cap >= 0 || throw(ArgumentError("centrality_correction!: cap must be nonnegative"))
  # f(λ) = λ + max(clamp(λ) − λ, −cap): the map applied to the eigenvalues
  # of w; the difference with w is then formed in place. Doing it this
  # way (rather than mapping λ ↦ max(clamp(λ) − λ, −cap) directly) keeps
  # the exact-copy shortcut of spectral_map! for a block already in the
  # box, which then contributes an exactly zero correction.
  spectral_map!(Δw, w, λ -> λ + max(clamp(λ, lo, hi) - λ, -cap), cone_dims)
  Δw .-= w
  return Δw
end
