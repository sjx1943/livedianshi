import React from 'react';
import { useTranslation } from 'react-i18next';

export default function TaskProgressGuidance({ feedback, taskTitle, ready, onConfirm }) {
  const { t } = useTranslation();
  const blocker = ready ? 'confirmation' : feedback?.completion_blocker || 'unknown';
  const unavailable = blocker === 'readiness_unavailable' || blocker === 'evaluation_unavailable';
  return (
    <aside data-testid="task-progress-guidance" tabIndex={0} className="px-4 py-3 text-sm border-b border-border bg-card text-foreground break-words shrink-0 max-h-[30vh] overflow-y-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
      <p role="status" className="font-semibold">{t(`task_progress_help_${blocker}`)}</p>
      {!ready && !unavailable && (
        <>
          {feedback?.reason && <p className="mt-1">{feedback.reason}</p>}
          <p className="mt-1">
            <span className="font-semibold">{t('task_progress_help_next')}</span>{' '}
            {feedback?.practice_tip || t('task_progress_help_practice', { task: taskTitle })}
          </p>
        </>
      )}
      {ready && <button type="button" onClick={onConfirm} className="mt-2 min-h-11 px-3 rounded-lg bg-primary text-white focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2">{t('task_progress_help_confirm')}</button>}
    </aside>
  );
}
