import type { ReactNode } from "react";

export function Panel({
  label,
  children,
  className = "",
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel px-5 py-5 pt-6 ${className}`}>
      <span className="panel-label">{label}</span>
      {children}
    </section>
  );
}
