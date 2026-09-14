const COMPLETION_SCORE = 9;
const MAX_PROGRESS_PER_TURN = 33;

/**
 * Translate the scoring contract into user-facing progress.
 *
 * Only completed scoring windows reach this function. A window may contain
 * three or four dialogue turns, so raw interaction_count is not a valid cap.
 * One applied window must never advance the bar by more than 33 percentage
 * points. A task remains at 99% until completion is explicitly confirmed.
 */
export function calculateTaskProgress({
  score,
  completedWindowCount,
  taskCompleted = false,
  previousProgress = null,
}) {
  if (taskCompleted) return 100;

  const safeScore = Math.max(0, Number(score) || 0);
  const scoreProgress = Math.min(100, Math.round((safeScore / COMPLETION_SCORE) * 100));
  const hasWindowCount = completedWindowCount !== null && completedWindowCount !== undefined;
  const safeWindowCount = Number(completedWindowCount);
  let progress = scoreProgress;
  if (hasWindowCount && Number.isFinite(safeWindowCount) && safeWindowCount >= 0) {
    progress = Math.min(progress, safeWindowCount * MAX_PROGRESS_PER_TURN);
  }

  if (previousProgress !== null && previousProgress !== undefined) {
    const previous = Math.max(0, Math.min(99, Number(previousProgress) || 0));
    progress = Math.min(progress, previous + MAX_PROGRESS_PER_TURN);
    // A delayed or duplicate backend snapshot must not move the same task
    // backwards. Task switches reset previousProgress before calling us.
    progress = Math.max(previous, progress);
  }

  return Math.max(0, Math.min(99, progress));
}

export function isCompletedWindowEvaluation(payload = {}) {
  return payload.evaluation_status === 'completed' || payload.window_completed === true;
}

// Completion/feedback messages can arrive after a task switch or reset.
export function isCurrentScoringMessage(payload, activeTask, generations, latestOrder = 0) {
  if (!payload?.task_id || !activeTask?.id || String(payload.task_id) !== String(activeTask.id)) return false;
  const expected = generations.get(String(payload.task_id)) ?? Number(activeTask.scoring_generation || 0);
  if (Number(payload.scoring_generation ?? 0) !== expected) return false;
  const baseline = expected === Number(activeTask.scoring_generation || 0)
    ? Number(activeTask.interaction_count || 0) : 0;
  return payload.interaction_count == null || Number(payload.interaction_count) >= Math.max(latestOrder, baseline);
}

export { COMPLETION_SCORE, MAX_PROGRESS_PER_TURN };
