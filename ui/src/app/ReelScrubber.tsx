/* ReelScrubber.tsx — the iPhone-style filmstrip used to pick each phase.
 *
 * A horizontally scrollable strip of real video thumbnails. A fixed marker at
 * the horizontal centre of the viewport is the capture window; the person drags
 * the film underneath it, exactly like scrubbing in iOS Photos. Scroll position
 * IS the value — there is no separate state to keep in sync — which is what
 * makes dragging feel native.
 *
 * Three things this has to get right:
 *
 *  - Cheap scrolling. The thumbnails are their own memoised component, so a
 *    scroll never re-renders up to 90 <img> elements, and the value is reported
 *    to the parent at most once per animation frame instead of once per scroll
 *    event. Without both, dragging stutters badly.
 *  - Following playback. While the video plays, the film scrolls to keep the
 *    playhead under the marker, without that programmatic scroll being mistaken
 *    for the user dragging.
 *  - Showing the other phases. Every phase's window is drawn on the strip, so
 *    when you move to Contact you can still see where Plant was set.
 */

import { memo, useCallback, useEffect, useLayoutEffect, useRef } from "react";

export interface ReelMarker {
  key: string;
  label: string;
  color: string;
  start: number;
  end: number;
  active: boolean;
  /** False until the user has actually visited and set this phase. */
  set: boolean;
}

/** The filmstrip itself. Memoised: it depends only on the images and geometry,
 *  never on the scroll position, so dragging does not re-render it. */
const Filmstrip = memo(function Filmstrip({
  thumbnails,
  filmWidth,
  padLeft,
  padRight,
}: {
  thumbnails: string[];
  filmWidth: number;
  padLeft: number;
  padRight: number;
}) {
  const count = thumbnails.length || 1;
  const thumbWidth = filmWidth / count;
  return (
    <div
      className="flex h-20 rounded-lg overflow-hidden bg-muted"
      style={{ width: filmWidth + padLeft + padRight, paddingLeft: padLeft, paddingRight: padRight }}
    >
      {thumbnails.length === 0 ? (
        <div style={{ width: filmWidth }} className="h-full flex items-center justify-center text-muted-foreground text-xs">
          Loading frames…
        </div>
      ) : (
        thumbnails.map((src, i) =>
          src ? (
            <img
              key={i}
              src={src}
              draggable={false}
              alt=""
              className="h-full object-cover shrink-0 pointer-events-none"
              style={{ width: thumbWidth }}
            />
          ) : (
            <div key={i} className="h-full shrink-0 bg-foreground/5 animate-pulse" style={{ width: thumbWidth }} />
          )
        )
      )}
    </div>
  );
});

export function ReelScrubber({
  thumbnails,
  duration,
  pxPerSecond,
  value,
  onChange,
  color,
  windowSeconds,
  markers,
  playheadTime,
  containerWidth,
  onMeasure,
}: {
  thumbnails: string[];
  duration: number;
  pxPerSecond: number;
  value: number;
  onChange: (t: number) => void;
  color: string;
  windowSeconds: number;
  markers: ReelMarker[];
  /** Non-null while the video is playing: the film follows it. */
  playheadTime: number | null;
  containerWidth: number;
  onMeasure: (width: number) => void;
}) {
  const outerRef = useRef<HTMLDivElement>(null);
  const suppressScroll = useRef(false);
  const dragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartScroll = useRef(0);
  const rafPending = useRef(false);
  // The value this component last reported or was told to show. Lets us skip
  // redundant scroll writes, which would otherwise fight the user mid-drag.
  const committed = useRef(value);

  const maxStart = Math.max(0, duration - windowSeconds);
  const windowWidthPx = Math.max(windowSeconds * pxPerSecond, 8);
  const filmWidth = Math.max(duration * pxPerSecond, 1);
  const padLeft = containerWidth / 2;
  const padRight = Math.max(containerWidth / 2 - windowWidthPx, 0);

  useLayoutEffect(() => {
    if (outerRef.current) onMeasure(outerRef.current.clientWidth);
  }, [onMeasure]);

  useEffect(() => {
    const onResize = () => {
      if (outerRef.current) onMeasure(outerRef.current.clientWidth);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [onMeasure]);

  /** Scroll the film to `t` without that being read back as a user drag. */
  const scrollToTime = useCallback((t: number) => {
    const el = outerRef.current;
    if (!el) return;
    suppressScroll.current = true;
    el.scrollLeft = t * pxPerSecond;
    committed.current = t;
    requestAnimationFrame(() => { suppressScroll.current = false; });
  }, [pxPerSecond]);

  // Position on mount and whenever the layout that defines the mapping changes
  // (viewport width, zoom level, window length).
  useLayoutEffect(() => {
    if (containerWidth === 0) return;
    scrollToTime(value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerWidth, pxPerSecond, windowSeconds]);

  // Follow playback. Skipped while the user is dragging — their hand wins.
  useEffect(() => {
    if (playheadTime === null || dragging.current || containerWidth === 0) return;
    if (Math.abs(playheadTime - committed.current) < 0.005) return;
    scrollToTime(Math.min(playheadTime, maxStart));
  }, [playheadTime, containerWidth, maxStart, scrollToTime]);

  // Externally driven value changes (phase switch, Reset, arrow keys).
  useEffect(() => {
    if (dragging.current || playheadTime !== null || containerWidth === 0) return;
    if (Math.abs(value - committed.current) < 1e-4) return;
    scrollToTime(value);
  }, [value, playheadTime, containerWidth, scrollToTime]);

  // Scroll events fire far faster than we can usefully do anything with them,
  // and each one that reaches React re-renders the page and seeks the video.
  // Collapsing to one per frame is what makes dragging feel immediate rather
  // than syrupy — the browser is already painting the film at native speed.
  const handleScroll = () => {
    if (suppressScroll.current || rafPending.current) return;
    rafPending.current = true;
    requestAnimationFrame(() => {
      rafPending.current = false;
      const el = outerRef.current;
      if (!el) return;
      const t = Math.min(Math.max(el.scrollLeft / pxPerSecond, 0), maxStart);
      if (Math.abs(t - committed.current) < 1e-4) return;
      committed.current = t;
      onChange(t);
    });
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!outerRef.current) return;
    dragging.current = true;
    dragStartX.current = e.clientX;
    dragStartScroll.current = outerRef.current.scrollLeft;
    outerRef.current.setPointerCapture(e.pointerId);
  };
  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current || !outerRef.current) return;
    outerRef.current.scrollLeft = dragStartScroll.current - (e.clientX - dragStartX.current);
  };
  const endDrag = () => { dragging.current = false; };

  return (
    // overflow-hidden here (not just on the scroll child below, which only
    // clips the X axis) contains the active-window spotlight's box-shadow —
    // its 9999px spread is what paints the dimmed area outside the capture
    // window, and with nothing clipping the Y axis that shadow bled past the
    // ~80px-tall strip and darkened the entire page above and below it.
    <div className="relative select-none overflow-hidden">
      <div
        ref={outerRef}
        onScroll={handleScroll}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
        onPointerCancel={endDrag}
        className="no-scrollbar overflow-x-scroll cursor-grab active:cursor-grabbing"
        style={{ WebkitOverflowScrolling: "touch", touchAction: "pan-x" }}
      >
        <div className="relative" style={{ width: filmWidth + padLeft + padRight }}>
          <Filmstrip
            thumbnails={thumbnails}
            filmWidth={filmWidth}
            padLeft={padLeft}
            padRight={padRight}
          />

          {/* Every phase's window, drawn on the film so the ones you already
              set stay visible while you work on the next. The active phase is
              handled by the fixed centre marker below, so it is skipped here. */}
          {markers.filter(m => !m.active && m.set).map(m => (
            <div
              key={m.key}
              className="absolute top-0 h-20 rounded-md pointer-events-none"
              style={{
                left: padLeft + m.start * pxPerSecond,
                width: Math.max((m.end - m.start) * pxPerSecond, 4),
                border: `2px solid ${m.color}`,
                backgroundColor: `${m.color}22`,
              }}
            >
              <span
                className="absolute -top-0.5 left-0 px-1 rounded-br text-[9px] font-black tracking-wider whitespace-nowrap"
                style={{ backgroundColor: m.color, color: "#07090a", fontFamily: "'Barlow Condensed', sans-serif" }}
              >
                {m.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Active window: fixed at the centre, the film moves under it. */}
      {containerWidth > 0 && (
        <>
          <div
            className="absolute top-0 h-20 pointer-events-none rounded-md"
            style={{
              left: containerWidth / 2,
              width: windowWidthPx,
              boxShadow: `0 0 0 2px ${color}, 0 0 0 9999px rgba(7,9,10,0.55)`,
            }}
          />
          <div
            className="absolute top-0 bottom-0 w-px bg-foreground/60 pointer-events-none"
            style={{ left: containerWidth / 2 }}
          />
        </>
      )}
    </div>
  );
}
