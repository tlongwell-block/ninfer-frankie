#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <iostream>
#include <stdexcept>

namespace q = ninfer::targets::qwen3_6;

q::PreparedPromptData prompt() {
    q::PreparedPromptData result;
    result.token_ids = {1, 2, 2, 3};
    result.token_types = {0, 0, 0, 0};
    result.positions = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    q::EmbeddingRowIdentity first{1, 5120, {}}, second{2, 5120, {}};
    first.digest.fill(11);
    second.digest.fill(22);
    result.embedding_identity = {first, second};
    return result;
}

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    auto original = prompt();
    q::detail::ResidentPrefixIdentity resident;
    q::detail::PrefixShortlistDigests shortlist;
    resident.assign(original);
    shortlist.assign(original);
    require(resident.matches(original, 4), "identical audio rows must match");
    auto revised = prompt();
    revised.embedding_identity[1].digest.fill(23);
    q::detail::PrefixShortlistDigests revised_shortlist;
    revised_shortlist.assign(revised);
    require(resident.matches(revised, 2), "changing an uncommitted audio suffix invalidated its prefix");
    require(!resident.matches(revised, 3), "same placeholder tokens reused changed audio");
    require(shortlist.at(2) == revised_shortlist.at(2), "future audio changed prefix shortlist");
    require(shortlist.at(3) != revised_shortlist.at(3), "changed audio retained shortlist identity");
    auto text = prompt();
    text.embedding_identity.clear();
    require(!resident.matches(text, 2), "text lookup incorrectly reused audio state");
    auto wrong_width = prompt();
    wrong_width.embedding_identity[0].width = 2048;
    require(!resident.matches(wrong_width, 2), "different embedding geometry matched");
    resident.truncate(2);
    shortlist.truncate(2);
    resident.append_generated(1, 0);
    shortlist.append_generated(std::vector<ninfer::TokenId>{9}, 0);
    auto rebuilt = prompt();
    rebuilt.token_ids = {1, 2, 9};
    rebuilt.token_types.resize(3);
    rebuilt.positions = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    rebuilt.embedding_identity.resize(1);
    q::detail::PrefixShortlistDigests rebuilt_shortlist;
    rebuilt_shortlist.assign(rebuilt);
    require(resident.matches(rebuilt, 3), "truncation retained an abandoned audio row");
    require(shortlist.at(3) == rebuilt_shortlist.at(3), "append after audio rollback changed identity");
    std::cout << "embedded prompt identity: passed\n";
}
