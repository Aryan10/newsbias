export default function SkippedSpecialist({ reason }: { reason?: string }) {
  return <div className="flex flex-col gap-2 pt-2">
    <p className="text-lg font-semibold text-slate-500 dark:text-slate-400">Not evaluated</p>
    <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
      {reason || "This specialist does not support the detected article language."}
    </p>
  </div>;
}
