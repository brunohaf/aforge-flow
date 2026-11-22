import { useCallback, useState } from "react";

// Theme is applied to <html> pre-paint in index.html; this hook toggles it.
export function useTheme() {
  const [dark, setDark] = useState<boolean>(() =>
    document.documentElement.classList.contains("dark")
  );

  const toggle = useCallback(() => {
    setDark((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem("aforge-theme", next ? "dark" : "light");
      return next;
    });
  }, []);

  return { dark, toggle };
}
