"use client";

import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("job-engine-theme", next ? "dark" : "light");
  }

  return (
    <button type="button" className="btn-theme" onClick={toggle}>
      dark mode
    </button>
  );
}
