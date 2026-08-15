/* scoring.js — thresholds, weights and coaching text.
 *
 * Every rule below is data, not logic: change `ideal`, `tol` or `weight` to
 * retune the model without touching any other file. Scores are a transparent
 * function of the measurement — there are no free-floating numbers.
 */

const VIS_GATE = 0.4;      // below this the landmark is too occluded to judge
const MIN_COVERAGE = 0.35; // below this share of a phase's weight, publish no score

/**
 * Map a measurement to 0-100.
 * Inside [ideal.lo, ideal.hi] -> 100. Beyond it, the score falls linearly and
 * reaches 40 once the value is `tol` outside the band, floored at 20.
 */
export function band(value, ideal, tol) {
  const [lo, hi] = ideal;
  if (value >= lo && value <= hi) return 100;
  const d = value < lo ? lo - value : value - hi;
  return Math.max(20, Math.round(100 - 60 * (d / tol)));
}

// Each formatter also knows how to render its ideal range, so the unit is not
// repeated twice ("0.30 × torso · ideal 0.05 to 0.40 × torso").
const deg = v => `${v.toFixed(0)}°`;
deg.range = (a, b) => `${a.toFixed(0)} to ${b.toFixed(0)}°`;
const tor = v => `${v.toFixed(2)} × torso`;
tor.range = (a, b) => `${a.toFixed(2)} to ${b.toFixed(2)} × torso`;
const rat = v => v.toFixed(2);
rat.range = (a, b) => `${a.toFixed(2)} to ${b.toFixed(2)}`;

/* ── Rules ──────────────────────────────────────────────── */

export const RULES = {
  plant: {
    label: 'Plant + Backswing',
    weight: 0.35,
    metrics: [
      {
        id: 'plantFootPlacement', label: 'Plant foot placement', weight: 1.2,
        ideal: [-0.05, 0.40], tol: 0.40, fmt: tor,
        good: 'Your support foot lands roughly level with your hips — a solid base to strike from.',
        low: {
          what: 'Your support foot is planted behind your hips.',
          why: 'Planting behind your body pushes your weight backwards, so you end up reaching for the ball and lifting it.',
          tip: 'Take a slightly longer final stride so the support foot arrives alongside the ball, not trailing it.'
        },
        high: {
          what: 'Your support foot is well ahead of your hips.',
          why: 'Over-striding stalls your momentum and forces the kicking leg to catch up, which costs power.',
          tip: 'Shorten the last step a little so you arrive balanced rather than braking.'
        }
      },
      {
        id: 'plantKneeBend', label: 'Plant-leg loading', weight: 1,
        ideal: [148, 172], tol: 22, fmt: deg,
        good: 'Your support knee is softly flexed — that absorbs the landing and keeps you stable.',
        low: {
          what: 'Your support knee is bent deeply.',
          why: 'Sinking too low drops your hips and makes it hard to hold the base while the kicking leg accelerates.',
          tip: 'Land taller. Think "quiet, springy knee" rather than sitting into the plant.'
        },
        high: {
          what: 'Your support leg is almost locked straight.',
          why: 'A locked knee cannot absorb the landing, so the impact goes into your hip and your base wobbles.',
          tip: 'Allow a small amount of bend as the foot lands.'
        }
      },
      {
        id: 'backswingKneeFlex', label: 'Backswing knee cock', weight: 1.1,
        ideal: [55, 110], tol: 35, fmt: deg,
        good: 'Your kicking knee is well cocked — the heel is loaded toward your glutes ready to whip through.',
        low: {
          what: 'Your kicking knee is folded very tightly.',
          why: 'An extreme fold usually means the leg is being pulled up rather than swung back, which shortens the strike path.',
          tip: 'Let the whole leg swing back from the hip instead of snapping the heel up.'
        },
        high: {
          what: 'You are taking the leg back nearly straight.',
          why: 'Power in a kick comes from unfolding the knee at the last moment. A straight-legged backswing removes that whip.',
          tip: 'Let the heel travel up toward your backside on the way back, then snap the lower leg through.'
        }
      },
      {
        id: 'backswingReach', label: 'Backswing length', weight: 1,
        ideal: [0.30, 1.05], tol: 0.40, fmt: tor,
        good: 'Good backswing length — enough runway to accelerate the leg into the ball.',
        low: {
          what: 'Your kicking foot has not travelled far behind your body.',
          why: 'A short backswing gives the leg very little distance to build speed, so the strike is mostly a push.',
          tip: 'Let the leg swing further back before you drive it forward — feel the hip open behind you.'
        },
        high: {
          what: 'Your backswing is unusually long.',
          why: 'An over-long swing is hard to time and often pulls your torso backwards as you strike.',
          tip: 'Trim the backswing slightly and focus on accelerating through the ball rather than winding up.'
        }
      },
      {
        id: 'balance', label: 'Body balance', weight: 1,
        ideal: [0, 0.35], tol: 0.35, fmt: tor,
        good: 'Your chest is stacked nicely over the support foot.',
        low: null,
        high: {
          what: 'Your upper body is offset from your support foot.',
          why: 'When the chest is not over the base, the plant leg fights to stay upright instead of anchoring the strike.',
          tip: 'Approach the ball a touch slower and let your chest arrive over the plant foot as it lands.'
        }
      },
      {
        id: 'torsoLean', label: 'Torso angle', weight: 0.9,
        ideal: [3, 28], tol: 20, fmt: deg,
        good: 'Slight forward lean into the plant — exactly what you want before striking.',
        low: {
          what: 'You are already leaning backwards at the plant.',
          why: 'Leaning back this early almost guarantees you get under the ball and skew the strike upward.',
          tip: 'Keep your chest travelling toward the target as the support foot lands.'
        },
        high: {
          what: 'You are leaning a long way forward.',
          why: 'Too much forward lean shortens the leg swing and can drag the strike low and across the ball.',
          tip: 'Stay tall through the plant; lean from the whole body, not by folding at the waist.'
        }
      }
    ]
  },

  contact: {
    label: 'Contact',
    weight: 0.40,
    metrics: [
      {
        id: 'torsoLean', label: 'Torso position', weight: 1.3,
        ideal: [3, 25], tol: 18, fmt: deg,
        good: 'Your chest is over the ball at contact — this is what keeps a shot down and driven.',
        low: {
          what: 'Your upper body is leaning backwards at contact.',
          why: 'A backward chest opens the strike face upward, which is the most common reason shots fly over the bar.',
          tip: 'Try keeping your chest slightly forward, over the ball, as your foot arrives.'
        },
        high: {
          what: 'You are folded well forward over the ball.',
          why: 'Excessive forward lean cramps the hip and cuts the leg swing short, so you lose power and often slice.',
          tip: 'Stay tall. Lean with your whole torso rather than bending at the waist.'
        }
      },
      {
        id: 'kickLegExtension', label: 'Kicking-leg extension', weight: 1.2,
        ideal: [138, 172], tol: 25, fmt: deg,
        good: 'Your kicking leg is extending strongly through contact without locking out.',
        low: {
          what: 'Your knee is still noticeably bent at contact.',
          why: 'Striking with a folded knee means the lower leg has not finished accelerating, so much of the available speed never reaches the ball.',
          tip: 'Let the shin snap through so the leg is close to straight as the foot meets the ball.'
        },
        high: {
          what: 'Your knee is locked out at contact.',
          why: 'A fully locked leg strikes with a rigid limb and offers no give, which is both weaker and harder on the joint.',
          tip: 'Keep a small amount of bend at impact and extend fully during the follow-through instead.'
        }
      },
      {
        id: 'plantFootPlacement', label: 'Plant foot position', weight: 1.1,
        ideal: [-0.10, 0.35], tol: 0.35, fmt: tor,
        good: 'Your support foot is positioned well relative to your body at contact.',
        low: {
          what: 'Your support foot sits behind your hips at contact.',
          why: 'With the plant behind you, your weight falls away from the ball and the strike loses its base.',
          tip: 'Aim to place the support foot level with the ball as you strike.'
        },
        high: {
          what: 'Your support foot is well ahead of your hips at contact.',
          why: 'Reaching past the ball means you strike behind your own base and have to pull the leg back through.',
          tip: 'Get the plant foot down beside the ball rather than past it.'
        }
      },
      {
        id: 'plantKneeBend', label: 'Plant-leg stability', weight: 0.9,
        ideal: [150, 176], tol: 22, fmt: deg,
        good: 'Your support leg is holding firm through the strike.',
        low: {
          what: 'Your support knee collapses as you strike.',
          why: 'If the base sinks during contact, energy leaks downward instead of into the ball.',
          tip: 'Brace the support leg — think of it as a post you swing around.'
        },
        high: {
          what: 'Your support leg is locked rigid.',
          why: 'A locked plant leg cannot absorb rotation, so your hips stall instead of turning through.',
          tip: 'Keep a hint of bend so the hip is free to rotate.'
        }
      },
      {
        id: 'balance', label: 'Body balance', weight: 1,
        ideal: [0, 0.30], tol: 0.30, fmt: tor,
        good: 'Well balanced — your mass is over your base at the moment of contact.',
        low: null,
        high: {
          what: 'Your upper body is off to one side of your support foot.',
          why: 'Striking off-balance means the leg has to correct mid-swing, which pulls the contact point around.',
          tip: 'Slow the last two steps and land with your chest over the plant foot.'
        }
      },
      {
        id: 'headOverBall', label: 'Head position', weight: 0.9,
        ideal: [-0.15, 0.35], tol: 0.35, fmt: tor,
        good: 'Your head is over your base — steady eyes and a stable strike.',
        low: {
          what: 'Your head is behind your support foot.',
          why: 'The head leads the torso. Hanging back tilts the whole body away from the ball.',
          tip: 'Keep your eyes down over the ball a moment longer through contact.'
        },
        high: {
          what: 'Your head is thrown well past your support foot.',
          why: 'Lunging the head forward drags your weight past the ball and shortens the strike.',
          tip: 'Let the leg travel to the ball; keep the head still.'
        }
      },
      {
        id: 'ankleLock', label: 'Ankle / foot lock', weight: 0.7, caveat: 'Foot landmarks are the noisiest part of the pose model — treat this as indicative.',
        ideal: [128, 170], tol: 30, fmt: deg,
        good: 'Your ankle looks firm and pointed, giving a solid striking surface.',
        low: {
          what: 'Your ankle appears relaxed / toes up at contact.',
          why: 'A loose ankle absorbs the impact instead of transferring it, and the strike surface moves on you.',
          tip: 'Point the toes down and lock the ankle before the foot arrives.'
        },
        high: {
          what: 'Your foot is extended very far in line with the shin.',
          why: 'Over-pointing can move contact onto the toes rather than the laces.',
          tip: 'Firm rather than maximal — aim to strike with the laces, ankle locked.'
        }
      },
      {
        id: 'hipRotation', label: 'Hip/shoulder separation', weight: 0.5, uncertain: true,
        caveat: 'A 2D proxy from apparent shoulder vs hip width. True 3D hip rotation cannot be measured from one camera.',
        ideal: [1.05, 1.55], tol: 0.45, fmt: rat,
        good: 'Shoulders appear to lead the hips, which is the pattern you want going into contact.',
        low: {
          what: 'Your shoulders and hips appear to be turning as one block.',
          why: 'Separation between the upper and lower body is what stores elastic energy for the strike.',
          tip: 'Let the shoulders open toward the target while the hips stay closed a fraction longer.'
        },
        high: {
          what: 'Your shoulders appear far more open than your hips.',
          why: 'This can mean the upper body is spinning out ahead of the strike, which pulls the contact across the ball.',
          tip: 'Drive the hip through with the leg instead of leading with the shoulder.'
        }
      }
    ]
  },

  followThrough: {
    label: 'Follow-through',
    weight: 0.25,
    metrics: [
      {
        id: 'kickLegExtension', label: 'Leg extension', weight: 1.2,
        ideal: [148, 180], tol: 25, fmt: deg,
        good: 'Full extension through the ball — the leg finished the job.',
        low: {
          what: 'Your kicking leg stays bent after contact.',
          why: 'Pulling the leg back early means you decelerated into the ball rather than through it.',
          tip: 'Let the leg keep straightening after contact instead of stopping at the ball.'
        },
        high: null
      },
      {
        id: 'followHeight', label: 'Follow-through height', weight: 1,
        ideal: [0.30, 1.20], tol: 0.40, fmt: tor,
        good: 'The kicking foot finishes high — a sign you swung through and not at the ball.',
        low: {
          what: 'Your follow-through is cut short and low.',
          why: 'A stopped follow-through almost always means you were slowing down before impact.',
          tip: 'Swing through the ball and let the momentum carry your foot up naturally.'
        },
        high: {
          what: 'Your foot finishes extremely high.',
          why: 'A very high finish often comes with leaning back, which sends the ball up.',
          tip: 'Keep the chest over the ball; the finish should be high but driven forward, not lifted.'
        }
      },
      {
        id: 'followReach', label: 'Follow-through direction', weight: 1,
        ideal: [0.25, 1.00], tol: 0.40, fmt: tor,
        good: 'The leg continues toward the target — energy went into the ball, not sideways.',
        low: {
          what: 'The kicking foot has not travelled far past your body.',
          why: 'A short forward path means the swing stopped at the ball or wrapped across it.',
          tip: 'Finish with the foot pointing where you want the ball to go.'
        },
        high: {
          what: 'The leg swings a very long way past your body.',
          why: 'An over-long finish can pull you off balance and delay your recovery.',
          tip: 'Let the swing decelerate naturally rather than throwing the leg.'
        }
      },
      {
        id: 'balance', label: 'Balance', weight: 1,
        ideal: [0, 0.45], tol: 0.40, fmt: tor,
        good: 'You stay balanced through the finish.',
        low: null,
        high: {
          what: 'Your upper body falls away from your support foot after contact.',
          why: 'Losing balance on the finish means the strike was not supported — and you cannot react to the rebound.',
          tip: 'Strengthen the plant leg and try to land the follow-through under control.'
        }
      },
      {
        id: 'torsoLean', label: 'Torso control', weight: 0.9,
        ideal: [0, 30], tol: 22, fmt: deg,
        good: 'Your torso stays over the strike as you finish.',
        low: {
          what: 'You fall backwards through the follow-through.',
          why: 'Falling away is the classic ball-over-the-bar pattern — the body opens up as the foot arrives.',
          tip: 'Keep the chest travelling toward the target after contact.'
        },
        high: {
          what: 'Your torso pitches sharply forward on the finish.',
          why: 'Folding forward after contact usually means you lunged rather than swung.',
          tip: 'Let the leg finish while the torso stays tall.'
        }
      },
      {
        id: 'support', label: 'Recovery / stability', weight: 0.9,
        ideal: [0, 0.50], tol: 0.40, fmt: tor,
        good: 'Your support foot is still under you — you could play the next action immediately.',
        low: null,
        high: {
          what: 'Your support foot ends up far from under your hips.',
          why: 'A base scattered away from your body means you need extra steps before you can react.',
          tip: 'Aim to land the follow-through and recover your base in one step.'
        }
      },
      {
        id: 'hipRotation', label: 'Hip rotation', weight: 0.5, uncertain: true,
        caveat: 'A 2D proxy from apparent shoulder vs hip width. True 3D hip rotation cannot be measured from one camera.',
        ideal: [0.95, 1.45], tol: 0.45, fmt: rat,
        good: 'Hips and shoulders appear to have rotated through together on the finish.',
        low: {
          what: 'Your hips appear not to have rotated through.',
          why: 'If the hip stays closed, the kick is powered by the leg alone rather than the whole body.',
          tip: 'Think about driving the kicking hip toward the target as the leg swings through.'
        },
        high: {
          what: 'Your shoulders appear to finish far ahead of your hips.',
          why: 'The upper body may be spinning off rather than rotating with the strike.',
          tip: 'Let hip and shoulder rotate together through the finish.'
        }
      }
    ]
  }
};

/* ── Engine ─────────────────────────────────────────────── */

/**
 * Score one measurement against its rule.
 * A metric is "uncertain" when the landmarks it depends on are occluded, when
 * the geometry is degenerate, or when the rule is flagged as a low-confidence
 * proxy. Uncertain metrics are reported but never counted in a score.
 */
function scoreMetric(rule, meas) {
  const out = {
    id: rule.id, label: rule.label, weight: rule.weight,
    value: meas?.value, vis: meas?.vis ?? 0,
    caveat: rule.caveat || null, ideal: rule.ideal, fmt: rule.fmt
  };

  if (!meas || !isFinite(meas.value)) {
    return { ...out, uncertain: true, reason: 'Could not measure this from the selected frame.' };
  }
  if (meas.vis < VIS_GATE) {
    return { ...out, uncertain: true, reason: 'The body parts this depends on are hidden or out of frame.' };
  }

  const score = band(meas.value, rule.ideal, rule.tol);
  const side = meas.value < rule.ideal[0] ? 'low' : (meas.value > rule.ideal[1] ? 'high' : 'good');
  const msg = side === 'good' ? { what: rule.good } : (rule[side] || { what: rule.good });

  return {
    ...out,
    score,
    uncertain: !!rule.uncertain,
    reason: rule.uncertain ? 'Low-confidence proxy — reported but not scored.' : null,
    feedback: msg
  };
}

const weightedMean = items => {
  const w = items.reduce((s, i) => s + i.weight, 0);
  return w > 0 ? Math.round(items.reduce((s, i) => s + i.score * i.weight, 0) / w) : null;
};

/** Score every phase and roll up to an overall figure. */
export function scoreAll(metrics) {
  const phases = {};
  for (const key of Object.keys(RULES)) {
    const rule = RULES[key];
    const scored = rule.metrics.map(r => scoreMetric(r, metrics[key][r.id]));
    const counted = scored.filter(s => !s.uncertain && s.score != null);
    // Coverage = share of this phase's intended weight we could actually measure.
    // A phase judged on a third of its metrics should not speak as loudly as a
    // fully measured one, so it is reported and then down-weighted accordingly.
    const totalW = rule.metrics.reduce((s, r) => s + r.weight, 0);
    const coverage = totalW > 0 ? counted.reduce((s, i) => s + i.weight, 0) / totalW : 0;
    // Below MIN_COVERAGE we refuse to publish a number at all. Reporting
    // "100/100" off a single measurable metric would be worse than saying
    // nothing, because it reads as a confident verdict.
    const insufficient = coverage < MIN_COVERAGE;
    phases[key] = {
      key, label: rule.label, weight: rule.weight,
      metrics: scored,
      score: insufficient ? null : weightedMean(counted),
      counted: counted.length,
      skipped: scored.length - counted.length,
      coverage,
      insufficient,
      partial: !insufficient && coverage < 0.6
    };
  }

  const usable = Object.values(phases).filter(p => p.score != null);
  const wOf = p => p.weight * p.coverage;
  const wSum = usable.reduce((s, p) => s + wOf(p), 0);
  const overall = wSum > 0
    ? Math.round(usable.reduce((s, p) => s + p.score * wOf(p), 0) / wSum)
    : null;

  return { phases, overall };
}

export const grade = s =>
  s == null ? '⚪' : s >= 85 ? '🟢' : s >= 70 ? '🟡' : '🔴';
