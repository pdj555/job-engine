"use client";

import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [light, setLight] = useState(false);

  useEffect(() => {
    setLight(document.documentElement.classList.contains("light"));
  }, []);

  function toggle() {
    const next = !light;
    setLight(next);
    document.documentElement.classList.toggle("light", next);
    localStorage.setItem("job-engine-theme", next ? "light" : "dark");
  }

  return (
    <button type="button" className="btn-theme" onClick={toggle}>
      {light ? "dark mode" : "light mode"}
    </button>
  );
}
