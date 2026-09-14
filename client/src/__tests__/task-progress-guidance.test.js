import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { I18nextProvider } from 'react-i18next';
import { createInstance } from 'i18next';
import zh from '../i18n/locales/zh.json';
import TaskProgressGuidance from '../components/TaskProgressGuidance';
import { isCurrentScoringMessage } from '../pages/conversationProgress';

test('reason and actionable tip survive the old toast timeout and update when ready', async () => {
  const i18n = createInstance();
  await i18n.init({ lng: 'zh', resources: { zh: { translation: zh } } });
  jest.useFakeTimers();
  const wrap = props => <I18nextProvider i18n={i18n}><TaskProgressGuidance taskTitle="点咖啡" {...props} /></I18nextProvider>;
  const { rerender } = render(wrap({ feedback: { completion_blocker: 'quality', reason: '缺少杯型', practice_tip: '请说：A small latte, please.' } }));
  act(() => jest.advanceTimersByTime(10000));
  expect(screen.getByText('缺少杯型')).toBeInTheDocument();
  expect(screen.getByText(/A small latte/)).toBeInTheDocument();
  rerender(wrap({ ready: true }));
  expect(screen.queryByText('缺少杯型')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: '确认完成' })).toBeInTheDocument();
  rerender(wrap({ feedback: { completion_blocker: 'readiness_unavailable' } }));
  expect(screen.getByRole('status')).toHaveTextContent('进度已保存');
  expect(screen.queryByText('下一步：')).not.toBeInTheDocument();
  jest.useRealTimers();
});

test('feedback rejects previous task, reset generation and out-of-order evaluations', () => {
  const task = { id: 372, scoring_generation: 1 };
  const generations = new Map([['372', 2]]);
  const fresh = { task_id: 372, scoring_generation: 2, interaction_count: 30 };
  expect(isCurrentScoringMessage(fresh, task, generations, 27)).toBe(true);
  expect(isCurrentScoringMessage({ ...fresh, task_id: 371 }, task, generations)).toBe(false);
  expect(isCurrentScoringMessage({ ...fresh, scoring_generation: 1 }, task, generations)).toBe(false);
  expect(isCurrentScoringMessage({ ...fresh, interaction_count: 24 }, task, generations, 27)).toBe(false);
});
