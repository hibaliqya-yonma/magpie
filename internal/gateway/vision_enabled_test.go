package gateway

import (
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/provider"
)

// routerUp is a router key whose vendor lists a small model that sees,
// anthropic/claude-haiku-5.5, which the user left unpicked: only "text",
// which can't see, is enabled (JonahFu on X: OpenRouter turned the unpicked
// Haiku away for their region, and they had never enabled it).
func routerUp(t *testing.T, picks ...string) (*Server, *eyeUp) {
	t.Helper()
	fresh(t)
	u := &eyeUp{}
	up := httptest.NewServer(u)
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: "router", Name: "Router", Chat: up.URL + "/v1", Key: "key", Models: picks}); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive("router", up.URL+"/v1", []catalog.Model{
		{ID: "text", ImageInput: imageInputBool(false), Price: &catalog.Price{Input: 0.1, Output: 0.1}},
		{ID: "anthropic/claude-haiku-5.5", Images: true, ImageInput: imageInputBool(true), Price: &catalog.Price{Input: 1, Output: 5}},
		{ID: "google/gemini-3-flash", Images: true, ImageInput: imageInputBool(true), Price: &catalog.Price{Input: 0.3, Output: 2.5}},
	}); err != nil {
		t.Fatal(err)
	}
	return New(), u
}

func TestAutoVisionPicksOnlyAnEnabledModel(t *testing.T) {
	s, u := routerUp(t, "text")
	if m := AutoVision(); m != "" {
		t.Fatalf("AutoVision = %q, a model the user never enabled", m)
	}
	code, body := postAs(t, s, "", `{"model":"router/text","messages":[{"role":"user","content":[{"type":"text","text":"read"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]}]}`)
	if code != 400 || !strings.Contains(body, "no enabled image recognition model is available") {
		t.Fatalf("%d %s", code, body)
	}
	if got := u.sent("anthropic/claude-haiku-5.5"); len(got) != 0 {
		t.Fatalf("the unpicked model was asked to describe: %v", got)
	}

	// with one that sees enabled, that one describes, not the cheaper
	// small model the user left out
	routerUp(t, "text", "anthropic/claude-haiku-5.5")
	if m := AutoVision(); m != "router/anthropic/claude-haiku-5.5" {
		t.Fatalf("AutoVision = %q", m)
	}
}

// A describer out of its allowance is passed over for one that can answer,
// resting after a refusal or read used up alike: a Claude account out of
// its five hours had each image sent to it, three by three, and refused.
// With none that can, no automatic describer is picked until one is back.
func TestAutoVisionPassesOverOneOut(t *testing.T) {
	fresh(t)
	for id, price := range map[string]float64{"cheap": 1, "dear": 5} {
		if err := provider.Save(provider.Provider{ID: id, Name: id, Chat: "http://" + id + ".invalid/v1", Key: id + "-key", Models: []string{"eye"}}); err != nil {
			t.Fatal(err)
		}
		if err := catalog.SaveLive(id, "http://"+id+".invalid/v1", []catalog.Model{
			{ID: "eye", Images: true, ImageInput: imageInputBool(true), Price: &catalog.Price{Input: price, Output: price}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	if m := AutoVision(); m != "cheap/eye" {
		t.Fatalf("AutoVision = %q, want the cheaper", m)
	}
	rest := func(ids ...string) {
		restingUntil.Lock()
		defer restingUntil.Unlock()
		for _, id := range ids {
			restingUntil.m[id] = time.Now().Add(time.Hour)
			restingUntil.note[id] = Rest{Why: failQuota, Status: 429, By: "window"}
		}
	}
	rest("cheap")
	if m := AutoVision(); m != "dear/eye" {
		t.Fatalf("AutoVision = %q with cheap out of quota", m)
	}
	rest("dear")
	if m := AutoVision(); m != "" {
		t.Fatalf("AutoVision = %q with every one out", m)
	}

	// used up as last read, before any refusal
	restingUntil.Lock()
	restingUntil.m = map[string]time.Time{}
	restingUntil.Unlock()
	windows := map[string]provider.Allowance{"cheap-key": week(100, time.Now().Add(time.Hour))}
	keyWindows(t, windows)
	if m := AutoVision(); m != "dear/eye" {
		t.Fatalf("AutoVision = %q with cheap's week used up", m)
	}
	windows["cheap-key"] = week(100, time.Now().Add(-time.Hour))
	if m := AutoVision(); m != "cheap/eye" {
		t.Fatalf("AutoVision = %q after cheap's week renewed", m)
	}

	// a quota rest for one model leaves the same provider's others ready
	rest("cheap/eye")
	p, err := provider.Find("cheap")
	if err != nil {
		t.Fatal(err)
	}
	p.Models = append(p.Models, "other-eye")
	if err := provider.Save(*p); err != nil {
		t.Fatal(err)
	}
	if err := catalog.SaveLive(p.ID, p.Chat, []catalog.Model{
		{ID: "eye", Images: true, Price: &catalog.Price{Input: 1, Output: 1}},
		{ID: "other-eye", Images: true, Price: &catalog.Price{Input: 2, Output: 2}},
	}); err != nil {
		t.Fatal(err)
	}
	if m := AutoVision(); m != "cheap/other-eye" {
		t.Fatalf("AutoVision = %q with only cheap/eye resting", m)
	}
}

// A known full allowance prevents the automatic helper's upstream call,
// through both an image-bearing chat request and the vision MCP tool.
func TestVisionDoesNotAskSpentDescriber(t *testing.T) {
	s, u := eyed(t)
	windows := map[string]provider.Allowance{"key": week(100, time.Now().Add(time.Hour))}
	keyWindows(t, windows)
	body := `{"model":"probe/text","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}}]}]}`
	code, out := postAs(t, s, "", body)
	if code != 400 || !strings.Contains(out, "no enabled image recognition model is available") {
		t.Fatalf("spent describer: %d %s", code, out)
	}
	r := httptest.NewRequest("POST", VisionMCPPath, nil)
	if _, err := s.visionMCPRun(r, "data:image/png;base64,aGVsbG8=", ""); err == nil || !strings.Contains(err.Error(), "no image recognition model available") {
		t.Fatalf("spent MCP describer: %v", err)
	}
	if n := len(u.sent("eye")); n != 0 {
		t.Fatalf("spent describer was asked %d times", n)
	}
	// no sticky disable: a renewed window lets the same image through
	windows["key"] = week(100, time.Now().Add(-time.Hour))
	if code, out := postAs(t, s, "", body); code != 200 || len(u.sent("eye")) != 1 {
		t.Fatalf("renewed describer: %d %s", code, out)
	}
}

// A subscription's full model pool does not consume another model's
// allowance, and a provider remains usable while a spare account can pay.
func TestVisionSpentSubscriptionAccounts(t *testing.T) {
	codexSignedIn(t, "spare@example.com")
	used := map[string]float64{"me@example.com": 100, "spare@example.com": 20}
	capUsage(t, used)
	p, err := provider.Find("codex")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	if visionSpent(*p, "gpt-6.1-sol", now) {
		t.Fatal("a spare account has allowance")
	}
	used["spare@example.com"] = 100
	if !visionSpent(*p, "gpt-6.1-sol", now) {
		t.Fatal("both accounts are out")
	}

	// Claude's shared five-hour window applies to Haiku as well as Opus.
	claude := provider.Provider{ID: "claude", Account: &provider.Account{Agent: "claude", User: "me@example.com"}}
	if !visionSpent(claude, "claude-haiku-5-5", now) {
		t.Fatal("Haiku was kept with Claude's shared window full")
	}
	allowances = func(string) map[string]provider.Allowance {
		return map[string]provider.Allowance{"me@example.com": {{Used: 100, Resets: now.Add(time.Hour), Model: "opus"}}}
	}
	if visionSpent(claude, "claude-haiku-5-5", now) {
		t.Fatal("Opus's pool should leave Haiku available")
	}
}

// The web-search side request's own pick is the same: a model the user
// enabled.
func TestSearcherModelIsAnEnabledModel(t *testing.T) {
	routerUp(t, "text")
	p, err := provider.Find("router")
	if err != nil {
		t.Fatal(err)
	}
	if m := searcherModel(*p); m != "text" {
		t.Fatalf("searcherModel = %q, want the only enabled model", m)
	}
	if picked := slices.ContainsFunc(p.Exposed(), func(m catalog.Model) bool { return m.ID == "google/gemini-3-flash" }); picked {
		t.Fatal("gemini is enabled")
	}
}
