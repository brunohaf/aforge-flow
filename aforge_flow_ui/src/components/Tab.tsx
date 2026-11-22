import { type ReactNode } from "react";

interface Props {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}

/** A small pill-style tab button. */
export default function Tab({ active, onClick, children }: Props) {
  return (
    <button
      onClick={onClick}
      className={
        "rounded px-3 py-1 text-sm " +
        (active ? "bg-indigo-600 text-white" : "bg-slate-100 dark:bg-slate-800")
      }
    >
      {children}
    </button>
  );
}
