import { render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { NetworkStore } from '../editor/networkStore.svelte';
import IssuesPanel from './IssuesPanel.svelte';

describe('IssuesPanel', () => {
  it('says the network is fine when there are no issues', () => {
    render(IssuesPanel, { props: { store: new NetworkStore() } });
    expect(screen.getByTestId('issues-empty')).toBeTruthy();
  });

  it('shows a warning with its fix', () => {
    const store = new NetworkStore();
    store.removeBlock(store.network.blocks[4].id);
    render(IssuesPanel, { props: { store } });

    const issue = screen.getByTestId('issue');
    expect(issue.textContent).toContain('Add a Softmax for probabilities');
    expect(issue.textContent).toContain('Add a Softmax block after the last Linear layer.');
    expect(screen.getByTestId('issue-fix')).toBeTruthy();
  });

  it('shows an error and selects the offending block when clicked', async () => {
    const store = new NetworkStore();
    store.updateBlock(store.network.blocks[0].id, { shape: [28, 28, 1] });
    render(IssuesPanel, { props: { store } });

    const issues = screen.getAllByTestId('issue');
    expect(issues.some((issue) => issue.textContent?.includes('Linear layer needs a flat list'))).toBe(
      true
    );

    await userEvent.click(issues[0]);
    expect(store.selectedBlockId).not.toBeNull();
  });
});
