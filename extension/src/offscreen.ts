import { InferenceEngine, listenForScoreRequests } from "./lib/engine";

listenForScoreRequests(new InferenceEngine(), "offscreen");
