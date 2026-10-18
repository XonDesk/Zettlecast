"""Tests for audio-checked corrections (acoustic scorer faked)."""

from unittest.mock import patch

import numpy as np

from zettlecast.podcast.enhancer import TranscriptEnhancer
from zettlecast.podcast.models import TranscriptSegment, WordTiming
from zettlecast.podcast.reference import RefWord
from zettlecast.podcast.verifier import Line, Suspect, TranscriptVerifier, Unit, _key


class FakeScorer:
    """NLL = sum of per-word costs; words not listed cost 0 (they 'fit the audio')."""

    def __init__(self, costs):
        self.costs = {k.lower(): v for k, v in costs.items()}
        self.calls = 0

    def nll(self, audio, texts):
        self.calls += 1
        return [sum(self.costs.get(_key(w), 0.0) for w in t.split()) for t in texts]


def segment(start, speaker, text):
    words = [WordTiming(text=w, start=start + i, end=start + i + 0.9) for i, w in enumerate(text.split())]
    return TranscriptSegment(start=start, end=words[-1].end, text=text, speaker=speaker, words=words)


SEGMENTS = [
    segment(0, "speaker_0", "I worked with the Norwegian triathletes Christian Blumfeld and Gustav Eden last year."),
    segment(20, "speaker_1", "Especially in trail learning um and ultra trail learning I think."),
]
AUDIO = np.zeros(16000 * 40, dtype=np.float32)
# what the publisher heard on line 1 ("trail running" where we heard "trail learning")
HEARD_LINE1 = [RefWord(20 + i, 20.9 + i, "Speaker 2", w) for i, w in enumerate(
    "Especially in trail running um and ultra trail running I think.".split())]


def verifier(costs=None, **kw):
    return TranscriptVerifier(SEGMENTS, AUDIO, FakeScorer(costs or {}), **kw)


def g(*terms, episode="earlier", source="speaker"):
    """A glossary whose terms were written down outside the pipeline in one earlier episode."""
    return {t: {episode: source} for t in terms}


def test_line_replace_merges_units_and_keeps_times():
    line = Line("[0.0s - 3.0s] s: ", [Unit("Gustav", 0, 1), Unit("Eden,", 1, 2), Unit("then", 2, 3)])
    line.replace(1, 2, ["Iden,"])
    assert line.content() == "Gustav Iden, then"
    assert [(u.start, u.end) for u in line.units] == [(0, 1), (1, 2), (2, 3)]
    line.replace(0, 2, ["Gustav", "Iden", "Jr,"])
    assert line.content() == "Gustav Iden Jr, then" and line.units[0].end == 2
    line.replace(3, 3, ["and"])  # insert before "then"
    assert line.content() == "Gustav Iden Jr, and then"
    line.replace(3, 4, [])  # delete
    assert line.content() == "Gustav Iden Jr, then"


def test_text_matches_transcript_format():
    v = verifier()
    assert v.text().split("\n")[0].startswith("[0.0s - 12.9s] speaker_0: I worked with")


def test_name_respelling_needs_a_confirmed_spelling():
    # a confident LLM respelling with nothing confirming the spelling goes to review
    v = verifier({"iden": 2.0, "ingebrigtsen": 40.0})
    corrections = v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden", "Gustav Ingebrigtsen"], "high")])
    assert [(c.chosen, c.confidence, c.alternatives) for c in corrections] == [("Gustav Eden", "low", ["Gustav Iden"])]
    assert "Gustav Eden last year." in v.text()
    # the show notes spell it: applied
    v = verifier({"iden": 2.0, "ingebrigtsen": 40.0})
    corrections = v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden", "Gustav Ingebrigtsen"], "high")],
                            vocabulary="Show notes: with Gustav Iden")
    assert [(c.original, c.chosen, c.confidence) for c in corrections] == [("Gustav Eden", "Gustav Iden", "high")]
    assert "Gustav Iden last year." in v.text()
    assert corrections[0].alternatives == []  # Ingebrigtsen didn't fit the audio


def test_inserting_a_name_where_words_were_heard_needs_the_audio_or_publisher():
    v = TranscriptVerifier([segment(0, "speaker_0", "so thanks all So see you next week")], AUDIO, FakeScorer({"jason": 2.0}))
    corrections = v.resolve([Suspect(0, "all So", ["Jason"], "high")], vocabulary="People: Jason Koop")
    assert all(c.chosen == c.original for c in corrections) and "thanks all So see" in v.text()


def test_name_that_does_not_fit_audio_is_never_applied():
    v = verifier({"ingebrigtsen": 40.0})
    assert v.resolve([Suspect(0, "Gustav Eden", ["Gustav Ingebrigtsen"], "high")]) == []
    assert "Gustav Eden" in v.text()


def test_low_confidence_guess_is_not_applied():
    v = verifier({"iden": 2.0})
    corrections = v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden"], "low")])
    assert all(c.confidence == "low" or c.chosen == c.original for c in corrections)
    assert "Gustav Eden" in v.text()


def test_publisher_hearing_a_change_backs_it():
    # other words than were recognised: the LLM alone isn't enough, the publisher hearing it is
    v = verifier({"running": 3.0})
    assert v.resolve([Suspect(1, "trail learning", ["trail running"], "medium")]) == []
    v = verifier({"running": 3.0}, reference=HEARD_LINE1)
    corrections = v.resolve([Suspect(1, "trail learning", ["trail running"], "medium")])
    assert [c.chosen for c in corrections] == ["trail running"]


def test_publisher_agreement_tolerates_word_time_offsets():
    # the publisher's words run half a word late, so "in" overlaps the span's start
    late = [RefWord(w.start + 0.5, w.end + 0.55, w.speaker, w.text) for w in HEARD_LINE1]
    v = verifier({"running": 3.0}, reference=late)
    assert [c.chosen for c in v.resolve([Suspect(1, "trail learning", ["trail running"], "medium")])] == ["trail running"]
    # the publisher heard the original too ("Coach" stays): no agreement for dropping it
    v = TranscriptVerifier([segment(0, "speaker_0", "with Coach Jason Koop today")], AUDIO, FakeScorer({"jason": 1.0}),
                           reference=[RefWord(i, i + 0.9, "S", w) for i, w in enumerate("with Coach Jason Koop today".split())])
    assert all(c.chosen == c.original for c in v.resolve([Suspect(0, "Coach Jason Koop", ["Jason Koop"], "high")]))


def test_publisher_hears_names_by_sound():
    # the publisher heard "Ryan": what was said is backed, the known spelling picks "Ryne"
    v = TranscriptVerifier([segment(0, "speaker_0", "thanks to Benat and Rind for coming on")], AUDIO,
                           FakeScorer({"ryne": 2.0, "ryan": 2.0}),
                           reference=[RefWord(i, i + 0.9, "S", w) for i, w in enumerate("thanks to Benat and Ryan for coming on".split())])
    corrections = v.resolve([Suspect(0, "Rind", ["Ryan"], "high")], vocabulary="People: Ryne Anderson")
    assert [c.chosen for c in corrections] == ["Ryne"]


def test_publisher_agreement_does_not_confirm_a_spelling():
    ref = [RefWord(10 + i * 0.5, 10.4 + i * 0.5, "Speaker 1", w) for i, w in enumerate(["Gustav", "Iden"])]
    v = verifier({"iden": 2.0}, reference=ref)
    corrections = v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden"], "medium")])
    assert all(c.chosen == c.original for c in corrections)


def test_glossary_term_corrects_without_the_llm():
    v = verifier({"blummenfelt": 3.0, "kristian": 1.0}, glossary=g("Kristian Blummenfelt"))
    corrections = v.resolve([])
    assert [(c.original, c.chosen, c.source) for c in corrections] == [
        ("Christian Blumfeld", "Kristian Blummenfelt", "glossary")
    ]


def test_cleanup_edits_checked_against_audio():
    v = verifier({"running": 3.0, "ingebrigtsen": 40.0}, reference=HEARD_LINE1)
    line0, line1 = v.text().split("\n")
    cleaned0 = line0.replace("Gustav Eden", "Gustav Ingebrigtsen")
    cleaned1 = line1.replace("trail learning um and ultra trail learning", "trail running and ultra-trail running")
    out, rejected = v.check_cleanup(0, f"{line0}\n{line1}", f"{cleaned0}\n{cleaned1}")
    assert "Gustav Eden" in out and "Ingebrigtsen" not in out
    assert "trail running and ultra-trail running I think." in out
    assert [r["suggested"] for r in rejected] == ["Ingebrigtsen"]
    # timings stay per word after case/punctuation-only edits
    assert v.lines[1].units[0].start == 20


def test_learn_only_names_written_down_outside_the_pipeline():
    v = verifier({"iden": 2.0})
    v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden"], "high")], vocabulary="Show notes: Gustav Iden")
    # an LLM correction and an LLM-inferred speaker name teach nothing
    assert v.learn("ep1", ["Jason Koop", "Adam"]) == {}
    # the publisher's speaker label does; so does a show-notes name the transcript has
    assert v.learn("ep1", ["Jason Koop"], verified=["Jason Koop"],
                   written="Guests: Sophie Herzog. We talk about Gustav Iden and the Norwegian method.") == {
        "Jason Koop": {"ep1": "speaker"}, "Gustav Iden": {"ep1": "show notes"}, "Norwegian": {"ep1": "show notes"},
    }


def test_learn_replaces_an_episodes_earlier_evidence():
    v = verifier(glossary={"Christian Blumfeld": {"ep1": "show notes", "ep0": "speaker"}, "Coconut": {"ep1": "show notes"}})
    expected = {"Christian Blumfeld": {"ep0": "speaker", "ep1": "show notes"}}
    assert v.learn("ep1", written="with Christian Blumfeld") == expected
    assert v.learn("ep1", written="with Christian Blumfeld") == expected  # reprocessing adds nothing


def test_show_notes_name_is_trusted_once_a_second_episode_confirms_it():
    v = verifier()
    v.learn("ep1", written="We talk about Christian Blumfeld. Buy the book.")
    assert v.trusted_terms() == []  # show notes have their own misspellings: one mention is not enough
    v.learn("ep2", written="More from Christian Blumfeld")
    assert v.trusted_terms() == ["Christian Blumfeld"]
    assert "Buy" not in v.glossary  # boilerplate, never capitalised mid-sentence in the transcript


def test_old_count_glossaries_are_not_evidence(tmp_path):
    from zettlecast.podcast.verifier import load_glossary, save_glossary

    path = tmp_path / "Show.json"
    path.write_text('{"Bennett": 38, "Benat Mencia": {"ep1": "speaker"}}', encoding="utf-8")
    assert load_glossary(path) == {"Benat Mencia": {"ep1": "speaker"}}
    save_glossary(path, {"A B": {"e1": "speaker"}, "C D": {"e1": "speaker", "e2": "speaker"}, "E F": {}})
    assert list(load_glossary(path)) == ["C D", "A B"]


def test_enhance_with_verifier_resolves_names_and_fixes_summary():
    v = verifier({"iden": 2.0})
    analysis = {"keywords": [], "summary": "She trained Gustav Eden.", "key_points": [], "chapters": [],
                "speakers": {"speaker_0": "Sophie Herzog"}}
    suspects = {"suspects": [{"line": 0, "text": "Gustav Eden", "candidates": ["Gustav Iden"], "confidence": "high"}]}
    with patch("zettlecast.podcast.enhancer.chat", side_effect=[analysis, suspects]):
        out = TranscriptEnhancer().enhance(v.text(), podcast="KoopCast", cleanup=False, verifier=v,
                                           show_notes="With Gustav Iden")
    assert "Sophie Herzog: I worked with" in out["cleaned_transcript"]
    assert "Gustav Iden last year." in out["cleaned_transcript"]
    assert out["summary"] == "She trained Gustav Iden."
    assert out["corrections"][0]["chosen"] == "Gustav Iden"
    assert out["verification"]["corrections"] == 1


def test_penalty_counts_only_changed_words():
    # "Gustav In" -> "Gustav Iversen": one word changed, so its whole cost counts
    v = TranscriptVerifier(
        [segment(0, "speaker_0", "the triathletes Christian Blumfeld and Gustav In and also")],
        AUDIO, FakeScorer({"iversen": 18.0}),
    )
    assert v.resolve([Suspect(0, "Gustav In", ["Gustav Iversen"], "high")]) == []


def test_name_part_and_show_notes_names():
    from zettlecast.podcast.verifier import _name_part, _spoken_names

    assert _name_part("Stephen Seiler has") == "Stephen Seiler"
    assert _name_part("said Stephen Seiler.") == "Stephen Seiler"
    assert _name_part("[[Bennett??]] said") == "Bennett"
    notes = "The Cocodona 250 recap. Links: https://Example.com/Koop More at KoopCast."
    assert _spoken_names(notes, "we ran the Cocodona 250 and KoopCast") == ["Cocodona 250", "KoopCast"]


def test_dropping_a_spoken_word_is_not_diluted():
    v = TranscriptVerifier([segment(0, "speaker_0", "work from Chris Ring's group showed that")],
                           AUDIO, FakeScorer({"ring": 12.0}))
    assert v.resolve([Suspect(0, "Chris Ring's group", ["Chris Ring"], "high")]) == []


def test_candidate_snaps_to_known_spelling():
    v = TranscriptVerifier([segment(0, "speaker_0", "thanks to Benat and Rind for coming on")],
                           AUDIO, FakeScorer({}))
    corrections = v.resolve([Suspect(0, "Rind", ["Ryan"], "high")], vocabulary="People: Ryne Anderson")
    assert [c.chosen for c in corrections] == ["Ryne"]


def test_learn_skips_sentence_starts_and_possessives():
    v = TranscriptVerifier([segment(0, "speaker_0", "now we go to the race and Koop's plan works")],
                           AUDIO, FakeScorer({}))
    from zettlecast.podcast.verifier import Correction

    v.corrections = [Correction(0, "So", "Now", "llm", 0, "high", looked_up=True),
                     Correction(1, "Coop's", "Koop's", "llm", 0, "high", looked_up=True)]
    assert v.learn("ep1", written="Now Koop's plan") == {"Koop": {"ep1": "wikipedia"}}



class FakeLookup:
    """Topic search results and linked pages, matched like the real lookup."""

    def __init__(self, results=(), linked=()):
        self.results, self.linked = set(results), set(linked)
        self.calls = []

    def find(self, heard, about):
        from zettlecast.podcast.lookup import best_matches

        self.calls.append((heard, about))
        return best_matches(heard, self.results), self.results | self.linked


def test_name_found_online_beats_a_confident_llm_guess():
    lookup = FakeLookup(results={"Kristian Blummenfelt", "Gustav Iden", "Casper Stornes"})
    v = verifier({"blummenfelt": 3.0, "blomfeldt": 2.0}, lookup=lookup)
    corrections = v.resolve([Suspect(0, "Christian Blumfeld", ["Christian Blomfeldt"], "high", "Norwegian triathlete")])
    assert [(c.chosen, c.source, c.confidence) for c in corrections] == [
        ("Kristian Blummenfelt", "wikipedia", "high")
    ]
    assert lookup.calls == [("Christian Blumfeld", "Norwegian triathlete")]
    assert v.learn("ep1") == {}  # proposed by the lookup alone: one source, not learned


def test_llm_name_confirmed_online_is_trusted():
    v = verifier({"iden": 2.0}, lookup=FakeLookup(linked={"Gustav Iden"}))
    corrections = v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden"], "medium", "Norwegian triathlete")])
    assert [(c.chosen, c.source) for c in corrections] == [("Gustav Iden", "llm+wikipedia")]
    assert v.learn("ep1") == {"Gustav Iden": {"ep1": "wikipedia"}}  # LLM + found online


def test_real_heard_name_is_kept_and_linked_pages_never_propose():
    # "Stephen Silver" sits in the linked pool; "Gustav Eden" itself is a real name there
    v = TranscriptVerifier([segment(0, "speaker_0", "the work of Stephen Seiler on polarised training")],
                           AUDIO, FakeScorer({}), lookup=FakeLookup(results={"Stephen Seiler"}, linked={"Stephen Silver"}))
    assert all(c.chosen == c.original for c in v.resolve(
        [Suspect(0, "Stephen Seiler", ["Steven Styler"], "high", "exercise physiologist")]))
    assert "Stephen Seiler on" in v.text()


def test_online_name_must_still_fit_the_audio():
    v = verifier({"blummenfelt": 30.0}, lookup=FakeLookup(results={"Kristian Blummenfelt"}))
    corrections = v.resolve([Suspect(0, "Christian Blumfeld", [], "high", "Norwegian triathlete")])
    assert corrections == [] and "Christian Blumfeld" in v.text()


def test_known_names_are_not_looked_up():
    lookup = FakeLookup(results={"Gustav Iden"})
    v = verifier({"iden": 2.0}, lookup=lookup, glossary=g("Gustav Iden"))
    v.resolve([Suspect(0, "Gustav Eden", ["Gustav Iden"], "high", "Norwegian triathlete")])
    v.resolve([Suspect(1, "trail learning", ["trail running"], "high", "")])
    assert lookup.calls == []


def test_lookup_proposes_only_close_multiword_names():
    from zettlecast.podcast.lookup import best_matches, name_titles

    names = name_titles({"Jim Walmsley (runner)", "Western States Endurance Run", "VO2 max",
                         "Kristian Blummenfelt", "Gustav (given name)", "Coca-Cola"})
    assert "Jim Walmsley" in names and "VO2 max" not in names
    assert best_matches("Christian Blunfeld", names) == ["Kristian Blummenfelt"]
    assert best_matches("Gustav In", names) == []  # different word count
    assert best_matches("Chimony", names) == []  # single words are never proposed
    assert best_matches("Kenji Niener", names) == []


def test_unchanged_name_is_not_credited_to_the_lookup():
    v = TranscriptVerifier([segment(0, "speaker_0", "we went through John and then home")],
                           AUDIO, FakeScorer({}), lookup=FakeLookup(linked={"John"}))
    corrections = v.resolve([Suspect(0, "through John", ["of John"], "high", "a person")])
    assert [(c.chosen, c.source) for c in corrections] == [("of John", "llm")]
    assert v.learn("ep1") == {}


def test_known_name_respelling_gets_a_looser_audio_limit():
    # "Ryne" is said "Ryan", but the recogniser prices the rare spelling ~16 higher
    v = TranscriptVerifier([segment(0, "speaker_0", "one of our coaches, Ryan Anderson, who you know")],
                           AUDIO, FakeScorer({"ryne": 16.0, "rhine": 16.0}))
    corrections = v.resolve([], vocabulary="People: Ryne Anderson")
    assert corrections == []  # not trusted yet: only the vocabulary, no glossary suspect
    v = TranscriptVerifier([segment(0, "speaker_0", "one of our coaches, Ryan Anderson, who you know")],
                           AUDIO, FakeScorer({"ryne": 16.0}), glossary=g("Ryne Anderson"))
    assert [c.chosen for c in v.resolve([], vocabulary="People: Ryne Anderson")] == ["Ryne Anderson,"]
    # a different, unconfirmed word over the normal limit is still rejected
    v = TranscriptVerifier([segment(0, "speaker_0", "one of our coaches, Ryan Anderson, who you know")],
                           AUDIO, FakeScorer({"rhine": 16.0}))
    assert v.resolve([Suspect(0, "Ryan Anderson", ["Rhine Anderson"], "high")]) == []


def test_cleanup_never_respells_a_confirmed_name():
    v = TranscriptVerifier([segment(0, "speaker_0", "thanks Ryne Anderson for coming")], AUDIO, FakeScorer({}),
                           glossary=g("Ryne Anderson"))
    v.resolve([])
    line = v.text()
    out, rejected = v.check_cleanup(0, line, line.replace("Ryne", "Ryan"))
    assert "Ryne Anderson" in out and rejected[0]["suggested"] == "Ryan"


def test_every_occurrence_in_a_line_is_checked():
    v = TranscriptVerifier([segment(0, "speaker_0", "with Benat and Ryan Anderson today. Ryan Anderson coaches")],
                           AUDIO, FakeScorer({}), glossary=g("Ryne Anderson"))
    v.resolve([])
    assert v.text().count("Ryne Anderson") == 2 and "Ryan" not in v.text()


def test_known_name_respelling_beats_publisher_agreeing_with_the_sound():
    ref = [RefWord(20 + i * 0.5, 20.4 + i * 0.5, "Speaker 1", w) for i, w in enumerate(["Ryan", "Anderson"])]
    v = TranscriptVerifier([segment(20, "speaker_0", "Ryan Anderson is here today")], AUDIO,
                           FakeScorer({"ryne": 16.0}), reference=ref, glossary=g("Ryne Anderson"))
    assert [c.chosen for c in v.resolve([])] == ["Ryne Anderson"]


def test_candidate_is_also_tried_in_a_known_spelling():
    v = TranscriptVerifier([segment(0, "speaker_0", "thanks Benette for that")], AUDIO, FakeScorer({}))
    corrections = v.resolve([Suspect(0, "Benette", ["Bennett"], "high")], vocabulary="People: Benat Mencia")
    assert [c.chosen for c in corrections] == ["Benat"]
