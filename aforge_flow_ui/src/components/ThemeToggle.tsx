import { useTheme } from "../hooks/useTheme";

/** Dark/light theme toggle button. */
export default function ThemeToggle() {
  const { dark, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      className="rounded border border-slate-300 px-2 py-1 text-sm dark:border-slate-600"
      title="theme"
    >
      {dark ? "🌙" : "☀️"}
    </button>
  );
}
