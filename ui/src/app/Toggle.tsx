/* Toggle.tsx — the switch used for the live-overlay options.
 *
 * Colours come from the theme tokens rather than literals, so these track the
 * rest of the site if the palette changes. `--switch-background` is the token
 * the shadcn theme already reserves for exactly this control.
 */

export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled = false,
  accent,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
  /** Overrides the "on" colour — used to tie a toggle to its phase colour. */
  accent?: string;
}) {
  const on = checked && !disabled;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`group flex items-center gap-3 w-full text-left rounded-lg px-3 py-2.5 border transition-all ${
        disabled
          ? "opacity-40 cursor-not-allowed border-border"
          : on
            ? "border-primary/40 bg-primary/5"
            : "border-border hover:border-border/80 hover:bg-muted/40"
      }`}
    >
      <span
        className="relative inline-flex shrink-0 items-center rounded-full transition-colors duration-200"
        style={{
          width: 38,
          height: 22,
          backgroundColor: on ? (accent ?? "var(--primary)") : "var(--switch-background)",
        }}
      >
        <span
          className="absolute rounded-full bg-white shadow-sm transition-transform duration-200"
          style={{
            width: 16,
            height: 16,
            left: 3,
            transform: `translateX(${on ? 16 : 0}px)`,
          }}
        />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-medium text-foreground truncate">{label}</span>
        {hint && <span className="block text-[11px] text-muted-foreground leading-snug">{hint}</span>}
      </span>
    </button>
  );
}
