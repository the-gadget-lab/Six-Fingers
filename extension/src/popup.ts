import { SettingsStore } from "./lib/settings";

class PopupView {
  constructor(private store: SettingsStore) {}

  async bind(): Promise<void> {
    const settings = await this.store.get();
    for (const key of ["enabled", "badgeAll", "blurAi"] as const) {
      const box = document.getElementById(key) as HTMLInputElement;
      box.checked = settings[key];
      box.addEventListener("change", () => void this.store.set({ [key]: box.checked }));
    }
  }
}

void new PopupView(new SettingsStore()).bind();
