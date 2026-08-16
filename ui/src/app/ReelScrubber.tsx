import { memo, useCallback, useEffect, useLayoutEffect, useRef } from "react";

export interface ReelMarker {
  key: string;
  label: string;
  color: string;
  start: number;
  end: number;
  active: boolean;
  
  set: boolean;
}


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

  
  const scrollToTime = useCallback((t: number) => {
    const el = outerRef.current;
    if (!el) return;
    suppressScroll.current = true;
    el.scrollLeft = t * pxPerSecond;
    committed.current = t;
    requestAnimationFrame(() => { suppressScroll.current = false; });
  }, [pxPerSecond]);

  
  useLayoutEffect(() => {
    if (containerWidth === 0) return;
    scrollToTime(value);
    
  }, [containerWidth, pxPerSecond, windowSeconds]);

  
  useEffect(() => {
    if (playheadTime === null || dragging.current || containerWidth === 0) return;
    if (Math.abs(playheadTime - committed.current) < 0.005) return;
    scrollToTime(Math.min(playheadTime, maxStart));
  }, [playheadTime, containerWidth, maxStart, scrollToTime]);

  
  useEffect(() => {
    if (dragging.current || playheadTime !== null || containerWidth === 0) return;
    if (Math.abs(value - committed.current) < 1e-4) return;
    scrollToTime(value);
  }, [value, playheadTime, containerWidth, scrollToTime]);

  
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
