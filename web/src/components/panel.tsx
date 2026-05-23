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
    <section className={`panel p-4 pt-5 ${className}`}>
      <span className="panel-label">{label}</span>
      {children}
    </section>
  );
}
