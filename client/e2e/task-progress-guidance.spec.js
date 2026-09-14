const { test, expect } = require('@playwright/test');

test('99% guidance persists and a qualified window can be confirmed @critical', async ({ page }, testInfo) => {
  const user = { id: 'guidance-test-user', username: 'Practice', native_language: 'zh', target_language: 'en' };
  let completed = false;
  await page.addInitScript(({ user }) => {
    localStorage.setItem('user', JSON.stringify(user));
    localStorage.setItem('ui_language', 'zh');
    localStorage.setItem('theme', 'light');
    localStorage.setItem('task_progress_Ordering%20Coffee', '99');
    window.testSockets = [];
    window.sentMessages = [];
    class Socket {
      static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
      constructor(url) {
        this.url = url; this.readyState = 0; this.bufferedAmount = 0;
        window.testSockets.push(this);
        setTimeout(() => { this.readyState = 1; this.onopen?.({}); }, 20);
      }
      send(data) { if (typeof data === 'string') window.sentMessages.push(JSON.parse(data)); }
      close() { this.readyState = 3; }
    }
    window.WebSocket = Socket;
  }, { user });
  await page.route(/tawk\.to|stripe\.com|dashscope|myqcloud|google/i, route => route.abort());
  await page.route('**/api/**', async route => {
    const url = route.request().url();
    const goal = { id: 1, target_language: 'en', status: 'active', scenarios: [{ title: 'Ordering Coffee', tasks: [{ id: 372, text: 'Order a latte', status: completed ? 'completed' : 'pending', score: 9, interaction_count: 27, scoring_generation: 0 }] }] };
    let data = {};
    if (url.includes('/users/profile')) data = { user };
    else if (url.includes('/v1/profile')) data = user;
    else if (url.includes('/goals/active')) data = { goal };
    else if (url.includes('/v1/goals')) data = [goal];
    else if (url.includes('/users/goals')) data = { goals: [goal] };
    else if (url.includes('/v1/tasks')) data = [{ id: 372, goal_id: 1, scenario_title: 'Ordering Coffee', task_description: 'Order a latte', status: completed ? 'completed' : 'pending', score: 9, interaction_count: 27, scoring_generation: 0 }];
    else if (url.includes('/v1/conversations') && route.request().method() === 'POST') data = { sessionId: 'guidance-test-session', id: 'guidance-test-session' };
    else if (url.includes('/history/') || url.includes('/v1/conversations')) data = [];
    else if (url.includes('/v1/realtime/tickets')) data = { ticket: 'test-ticket' };
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ success: true, data }) });
  });
  await page.goto('/conversation?scenario=Ordering%20Coffee');
  await expect(page.getByRole('list', { name: '场景子任务' })).toContainText('Order a latte');
  await expect.poll(() => page.evaluate(() => window.testSockets.some(s => s.readyState === 1 && s.url.includes('/realtime')))).toBe(true);
  const emit = (type, payload) => page.evaluate(({ type, payload }) => {
    window.testSockets.filter(s => s.readyState === 1 && s.url.includes('/realtime')).forEach(s => s.onmessage?.({ data: JSON.stringify({ type, payload }) }));
  }, { type, payload });
  await emit('session_restored', { task_id: 372, score: 9, interaction_count: 27, scoring_generation: 0 });
  const feedback = { task_id: 372, task_score: 9, interaction_count: 27, scoring_generation: 0, completed_window_count: 9, evaluation_status: 'completed', evaluation_id: 'window-9', delta: 1, quality: 'needs_work', completion_blocker: 'quality', reason: '还没有说明杯型。', practice_tip: '请补充杯型，例如：A small latte, please.', task_ready_to_complete: false };
  await emit('proficiency_update', feedback);
  const guidance = page.getByTestId('task-progress-guidance');
  await expect(guidance).toContainText('本轮表现还需改进');
  await page.waitForTimeout(3500); // Outlive the existing score-toast dismissal.
  await expect(guidance).toContainText('A small latte, please.');
  await emit('proficiency_update', { ...feedback, task_id: 371, evaluation_id: 'old-task', reason: 'WRONG TASK' });
  await emit('proficiency_update', { ...feedback, scoring_generation: -1, evaluation_id: 'old-generation', reason: 'OLD GENERATION' });
  await expect(guidance).not.toContainText(/WRONG TASK|OLD GENERATION/);
  await emit('scoring_feedback', { task_id: 372, scoring_generation: 0, interaction_count: 27, completion_blocker: 'readiness_unavailable' });
  await expect(guidance).toContainText('完成确认暂时不可用');
  await expect(guidance).not.toContainText('本轮表现还需改进');
  await emit('session_restored', { task_id: 372, score: 9, interaction_count: 27, scoring_generation: 0, progress_feedback: feedback });
  await expect(guidance).toContainText('A small latte, please.');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize().width + 1);
  await page.screenshot({ path: testInfo.outputPath('persistent-guidance.png') });
  // CC overlays the regular task panel; feedback must remain available there.
  const ccButton = page.getByRole('button', { name: /进入 CC/ });
  if (await ccButton.isVisible()) {
    await ccButton.click();
    await expect(guidance).toContainText('A small latte, please.');
    await page.screenshot({ path: testInfo.outputPath('cc-guidance.png') });
    await page.getByRole('button', { name: '退出 CC 沉浸模式' }).first().click();
  }
  await emit('proficiency_update', { ...feedback, interaction_count: 30, completed_window_count: 10, evaluation_id: 'window-10', quality: 'satisfactory', completion_blocker: null, task_ready_to_complete: true, ready_token: 'test-ready-token-1234567890' });
  await emit('task_ready_to_complete', { task_id: 372, task_title: 'Order a latte', scoring_generation: 0, interaction_count: 30, ready_token: 'test-ready-token-1234567890' });
  await expect(page.getByRole('dialog')).toBeVisible();
  await emit('session_restored', { task_id: 372, score: 9, interaction_count: 30, scoring_generation: 0, progress_feedback: { task_id: 372, scoring_generation: 0, interaction_count: 30, completion_blocker: 'readiness_unavailable' } });
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(guidance).toContainText('完成确认暂时不可用');
  await expect(guidance.getByRole('button', { name: '确认完成' })).toHaveCount(0);
  await emit('task_ready_to_complete', { task_id: 372, task_title: 'Order a latte', scoring_generation: 0, interaction_count: 30, ready_token: 'test-ready-token-1234567890' });
  await page.getByRole('dialog').getByRole('button', { name: '继续深入当前任务' }).click();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(guidance).toContainText('已达到完成条件');
  await guidance.getByRole('button', { name: '确认完成' }).click();
  await page.getByRole('dialog').getByRole('button', { name: /完成当前任务/ }).click();
  await expect.poll(() => page.evaluate(() => window.sentMessages.some(m => m.type === 'user_confirmed_complete' && m.payload.task_id === 372 && m.payload.ready_token === 'test-ready-token-1234567890'))).toBe(true);
  completed = true;
  await emit('task_completed', { task_id: 372, task_title: 'Order a latte', scoring_generation: 0, score: 9, next_task: null });
  await expect(guidance).toHaveCount(0);
  await expect(page.getByRole('progressbar', { name: '当前子任务进度', exact: true })).toHaveAttribute('aria-valuenow', '100');
});
