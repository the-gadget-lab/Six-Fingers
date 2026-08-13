export interface Settings {
  enabled: boolean;
  threshold: number;
  badgeAll: boolean;
  blurAi: boolean;
}

const DEFAULTS: Settings = { enabled: true, threshold: 0.65, badgeAll: true, blurAi: true };

export class SettingsStore {
  async get(): Promise<Settings> {
    const stored = await chrome.storage.local.get({ ...DEFAULTS });
    return stored as unknown as Settings;
  }

  async set(patch: Partial<Settings>): Promise<void> {
    await chrome.storage.local.set(patch);
  }

  onChange(fn: (s: Settings) => void): void {
    chrome.storage.local.onChanged.addListener(() => void this.get().then(fn));
  }
}
