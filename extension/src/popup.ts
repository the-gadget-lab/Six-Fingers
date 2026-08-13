import { SettingsStore } from "./lib/settings";

class PopupView {
  constructor(private store: SettingsStore) {}

  async bind(): Promise<void> {
    document.getElementById("ver")!.textContent = "v" + chrome.runtime.getManifest().version;
    const settings = await this.store.get();
    document.body.classList.toggle("off", !settings.enabled);
    for (const key of ["enabled", "badgeAll", "blurAi"] as const) {
      const box = document.getElementById(key) as HTMLInputElement;
      box.checked = settings[key];
      box.addEventListener("change", () => {
        void this.store.set({ [key]: box.checked });
        if (key === "enabled") document.body.classList.toggle("off", !box.checked);
      });
    }
  }
}

void new PopupView(new SettingsStore()).bind();
