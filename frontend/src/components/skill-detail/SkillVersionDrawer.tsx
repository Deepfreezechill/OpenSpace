import { useLayoutEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import type { SkillDetail } from '../../api';
import { parseDiff } from '../../utils/diffParser';
import EmptyState from '../EmptyState';
import ProgressBar from '../ProgressBar';
import DiffViewer from './DiffViewer';
import { formatDate, formatPercent, truncate } from '../../utils/format';

interface SkillVersionDrawerProps {
  skill: SkillDetail | null;
  isOpen: boolean;
  onClose: () => void;
}

const DRAWER_ANIMATION_DURATION_MS = 300;
const MAX_RENDERABLE_DIFF_LENGTH = 250_000;
const APP_ROOT_SELECTOR = '#root';
const SKILL_MD_FILENAME = 'SKILL.md';

function resolveSourcePreview(skill: SkillDetail) {
  const snapshot = skill.lineage.content_snapshot;
  if (snapshot && Object.prototype.hasOwnProperty.call(snapshot, SKILL_MD_FILENAME)) {
    return {
      path: `Version snapshot - ${SKILL_MD_FILENAME}`,
      content: snapshot[SKILL_MD_FILENAME] ?? '',
    };
  }

  if (skill.source?.exists && skill.source.content !== null) {
    return {
      path: skill.source.path || skill.path || SKILL_MD_FILENAME,
      content: skill.source.content,
    };
  }

  return null;
}

function lockScroll() {
  const html = document.documentElement;
  const body = document.body;
  const appRoot = document.querySelector<HTMLElement>(APP_ROOT_SELECTOR);
  const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const supportsStableScrollbarGutter = typeof CSS !== 'undefined' && CSS.supports?.('scrollbar-gutter: stable');

  const bodyScrollbarWidth = supportsStableScrollbarGutter ? 0 : Math.max(0, window.innerWidth - html.clientWidth);

  html.classList.add('drawer-open');
  body.classList.add('drawer-open');
  if (appRoot) {
    appRoot.inert = true;
    appRoot.setAttribute('aria-hidden', 'true');
  }
  if (bodyScrollbarWidth > 0) {
    body.style.paddingRight = `${bodyScrollbarWidth}px`;
  }

  return () => {
    html.classList.remove('drawer-open');
    body.classList.remove('drawer-open');
    body.style.removeProperty('padding-right');
    if (appRoot) {
      appRoot.inert = false;
      appRoot.removeAttribute('aria-hidden');
    }
    if (previouslyFocused && previouslyFocused !== body && previouslyFocused.isConnected) {
      previouslyFocused.focus({ preventScroll: true });
    }
  };
}

export default function SkillVersionDrawer({ skill, isOpen, onClose }: SkillVersionDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  const rawDiff = skill?.lineage.content_diff ?? '';
  const isOversizedDiff = rawDiff.length > MAX_RENDERABLE_DIFF_LENGTH;
  const diffFiles = useMemo(() => (isOversizedDiff ? [] : parseDiff(rawDiff)), [isOversizedDiff, rawDiff]);
  const canShowDiff = rawDiff.trim().length > 0;

  useLayoutEffect(() => {
    if (!skill) {
      return;
    }
    return lockScroll();
  }, [skill]);

  useLayoutEffect(() => {
    if (!isOpen) {
      return;
    }
    closeButtonRef.current?.focus();
  }, [isOpen]);

  useLayoutEffect(() => {
    if (!skill) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, skill]);

  if (!skill) {
    return null;
  }

  const sourcePreview = resolveSourcePreview(skill);

  const drawerContent = (
    <>
      <button
        type="button"
        aria-label="Close version detail"
        className={`fixed inset-0 z-30 bg-[rgba(20,20,19,0.22)] transition-opacity duration-300 ${isOpen ? 'pointer-events-auto opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="skill-version-drawer-title"
        className={`fixed top-0 right-0 z-40 flex h-full min-w-[28rem] max-w-[65vw] border-l-2 border-[color:var(--color-ink)] bg-[color:var(--color-surface)] shadow-lg will-change-transform ${isOpen ? 'pointer-events-auto' : 'pointer-events-none'}`}
        style={{
          width: '65vw',
          animation: `${isOpen ? 'drawer-slide-in' : 'drawer-slide-out'} ${DRAWER_ANIMATION_DURATION_MS}ms ease-in-out forwards`,
        }}
      >
        <div className="drawer-scroll flex h-full w-full flex-col overflow-hidden overscroll-contain">
          <header className="p-4 border-b-2 border-[color:var(--color-border)] flex items-start justify-between gap-3 shrink-0">
            <div className="min-w-0">
              <p className="text-xs uppercase tracking-wide text-muted">Skill Version</p>
              <h2 id="skill-version-drawer-title" className="font-bold text-lg truncate">{skill.name}</h2>
              <p className="text-xs text-muted font-mono break-all">{skill.skill_id}</p>
            </div>
            <div className="flex items-center gap-2">
              <Link to={`/skills/${encodeURIComponent(skill.skill_id)}`} className="btn-outline-ink text-sm">
                Open as main
              </Link>
              <button type="button" onClick={onClose} ref={closeButtonRef} className="btn-outline-ink text-sm">
                Close
              </button>
            </div>
          </header>

          <main className="drawer-scroll drawer-scroll-region flex-1 overflow-y-auto overscroll-contain space-y-4 bg-bg-page p-4">
            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-2 min-w-0">
                  <div className="text-xs uppercase tracking-[0.16em] text-muted">Version Summary</div>
                  <div className="text-sm text-muted">{skill.description || 'No description available for this version.'}</div>
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="tag px-2 py-1">{skill.category}</span>
                    <span className="tag px-2 py-1">{skill.origin}</span>
                    <span className="tag px-2 py-1">gen {skill.generation}</span>
                    <span className="tag px-2 py-1">{skill.is_active ? 'active' : 'inactive'}</span>
                    {skill.tags.map((tag) => (
                      <span key={tag} className="tag px-2 py-1">{tag}</span>
                    ))}
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <div className="text-4xl font-bold font-serif leading-none">{skill.score.toFixed(1)}</div>
                  <div className="text-xs uppercase tracking-[0.16em] text-muted mt-2">version score</div>
                </div>
              </div>
            </section>

            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 space-y-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-muted">Metrics</div>
                <h3 className="text-xl font-bold font-serif mt-1">Execution quality</h3>
              </div>
              <div className="space-y-4">
                <ProgressBar label="Effective rate" value={skill.effective_rate} colorClass="bg-primary" />
                <ProgressBar label="Completion rate" value={skill.completion_rate} colorClass="bg-accent" />
                <ProgressBar label="Applied rate" value={skill.applied_rate} colorClass="bg-teal" />
                <ProgressBar label="Fallback rate" value={skill.fallback_rate} colorClass="bg-danger" />
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm text-muted">
                <div><div className="font-bold text-ink">Selections</div><div>{skill.total_selections}</div></div>
                <div><div className="font-bold text-ink">Applied</div><div>{skill.total_applied}</div></div>
                <div><div className="font-bold text-ink">Completions</div><div>{skill.total_completions}</div></div>
                <div><div className="font-bold text-ink">Fallbacks</div><div>{skill.total_fallbacks}</div></div>
              </div>
            </section>

            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 text-sm space-y-2">
              <h3 className="font-bold">Version Metadata</h3>
              <p><strong>Origin:</strong> {skill.origin}</p>
              <p><strong>Generation:</strong> {skill.generation}</p>
              <p><strong>Visibility:</strong> {skill.visibility}</p>
              <p><strong>Created:</strong> {formatDate(skill.lineage.created_at)}</p>
              <p><strong>First seen:</strong> {formatDate(skill.first_seen)}</p>
              <p><strong>Last updated:</strong> {formatDate(skill.last_updated)}</p>
              <p><strong>Skill path:</strong> <span className="break-all">{skill.path || 'Unavailable'}</span></p>
              <p><strong>Skill dir:</strong> <span className="break-all">{skill.skill_dir || 'Unavailable'}</span></p>
              <p><strong>Parent IDs:</strong> {skill.parent_skill_ids.length ? skill.parent_skill_ids.join(', ') : 'None'}</p>
              <p><strong>Change summary:</strong> {skill.lineage.change_summary || 'None'}</p>
              <p><strong>Effective score:</strong> {formatPercent(skill.effective_rate)}</p>
            </section>

            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 space-y-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-muted">Diff</div>
                <h3 className="text-xl font-bold font-serif mt-1">Content diff</h3>
              </div>
              {isOversizedDiff ? (
                <EmptyState title="Diff too large" description="This version has a very large content diff, so the inline viewer is disabled." />
              ) : canShowDiff ? (
                diffFiles.length > 0 ? (
                  <DiffViewer files={diffFiles} />
                ) : (
                  <EmptyState title="Diff unavailable" description="This version has a diff payload, but it could not be parsed as a unified diff." />
                )
              ) : (
                <EmptyState title="No content diff" description="This version does not have a stored content diff." />
              )}
            </section>

            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 space-y-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-muted">Source</div>
                <h3 className="text-xl font-bold font-serif mt-1">SKILL.md preview</h3>
              </div>
              {sourcePreview ? (
                <div className="space-y-3">
                  <div className="text-xs text-muted break-all">{sourcePreview.path}</div>
                  <pre className="field-surface p-4 text-xs overflow-auto max-h-[320px] whitespace-pre-wrap">{sourcePreview.content}</pre>
                </div>
              ) : (
                <EmptyState title="Source unavailable" description="This version points to a missing or unreadable SKILL.md path." />
              )}
            </section>

            <section className="rounded-[var(--radius)] border-2 border-[color:var(--color-border-dark)] bg-surface p-4 space-y-4">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-muted">Analyses</div>
                <h3 className="text-xl font-bold font-serif mt-1">Recent execution analyses</h3>
              </div>
              {skill.recent_analyses.length > 0 ? (
                <div className="space-y-3">
                  {skill.recent_analyses.map((analysis) => (
                    <div key={`${analysis.task_id}-${analysis.timestamp}`} className="panel-subtle p-4 bg-surface space-y-2">
                      <div className="flex items-center justify-between gap-3">
                        <div className="font-bold truncate">{analysis.task_id}</div>
                        <div className="text-xs text-muted">{formatDate(analysis.timestamp)}</div>
                      </div>
                      <div className="text-sm text-muted">{truncate(analysis.execution_note || 'No execution note', 220)}</div>
                      <div className="text-xs text-muted">
                        completed: {analysis.task_completed ? 'yes' : 'no'} · tool issues: {analysis.tool_issues.length} · suggestions: {analysis.evolution_suggestions.length}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState title="No analyses yet" description="Execution analyses will appear after recorded task runs are persisted into SQLite." />
              )}
            </section>
          </main>
        </div>
      </aside>
    </>
  );

  if (typeof document === 'undefined') {
    return drawerContent;
  }

  return createPortal(drawerContent, document.body);
}
