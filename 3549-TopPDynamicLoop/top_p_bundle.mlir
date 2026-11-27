// Top-p candidates for sampling, with a loop whose bound the loop body updates.
// Example for RFC 3549 (3549-TopPDynamicLoopRFC.md); the op of each SDSC is in its file name and comment.
//
// V = 131072 vocab, B <= 32 rows (one stick here; more rows take more sticks), all tensors FP32 with stick
// dim B, a row position being one 128 B stick. G = 128 candidates per group, Gmax = V/G = 1024 groups at
// most (p reached only with every token), R = Gmax*G = V candidate rows.
//
// Inputs:  probs     [V, B]  softmax output, masked in place by the loop (consumed)
//          p         [1, B]  top-p threshold per row
//          zero, one, gmax [1, B]  precomputed constants: 0, 1 and Gmax in every lane
// Outputs: cand_prob [R, B]  kept probabilities, descending, 0 for every candidate not kept
//          cand_tok  [R, B]  token ids of the candidates, integers held in FP32
//          ngroups   [1, B]  groups holding a kept candidate, at most Gmax

// group g of a [R, B] tensor: G = 128 sticks of 128 B
#group = affine_map<(g)[base] -> (base + 16384 * g)>

module {
  func.func @top_p(%probs_arg: !sdscbundle.input_arg<index>, %p_arg: !sdscbundle.input_arg<index>,
                   %zero_arg: !sdscbundle.input_arg<index>, %one_arg: !sdscbundle.input_arg<index>,
                   %gmax_arg: !sdscbundle.input_arg<index>,
                   %cand_prob_arg: !sdscbundle.input_arg<index>, %cand_tok_arg: !sdscbundle.input_arg<index>,
                   %ngroups_arg: !sdscbundle.input_arg<index>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %probs = sdscbundle.input_arg_extract value from %probs_arg : !sdscbundle.input_arg<index> -> index
    %p = sdscbundle.input_arg_extract value from %p_arg : !sdscbundle.input_arg<index> -> index
    %zero = sdscbundle.input_arg_extract value from %zero_arg : !sdscbundle.input_arg<index> -> index
    %one = sdscbundle.input_arg_extract value from %one_arg : !sdscbundle.input_arg<index> -> index
    %gmax = sdscbundle.input_arg_extract value from %gmax_arg : !sdscbundle.input_arg<index> -> index
    %cand_prob = sdscbundle.input_arg_extract value from %cand_prob_arg : !sdscbundle.input_arg<index> -> index
    %cand_tok = sdscbundle.input_arg_extract value from %cand_tok_arg : !sdscbundle.input_arg<index> -> index
    %ngroups = sdscbundle.input_arg_extract value from %ngroups_arg : !sdscbundle.input_arg<index> -> index

    // Scratch: one-stick tensors [1, B], then two [G, B] tensors for the cut of a group.
    %pool = sdscbundle.device_mem_allocate 65536 bytes : index
    %acc = arith.addi %pool, %c0 : index             // probability taken so far
    %o_cont = arith.constant 128 : index
    %cont = arith.addi %pool, %o_cont : index        // 1 for a row still below p
    %o_t = arith.constant 256 : index
    %t = arith.addi %pool, %o_t : index
    %o_bound = arith.constant 384 : index
    %bound_addr = arith.addi %pool, %o_bound : index // the loop bound, first value of the stick
    %o_cs = arith.constant 16384 : index
    %cs = arith.addi %pool, %o_cs : index            // cumsum of the group [G, B]
    %o_cs_last = arith.constant 32640 : index
    %cs_last = arith.addi %pool, %o_cs_last : index  // its last row, G-1: the group's sum
    %o_before = arith.constant 32768 : index
    %before = arith.addi %pool, %o_before : index    // mass before each candidate [G, B], then the drop flags

    // ---- Before the loop: initial values, copied (identity) from the constants, broadcast over the rows
    // for cand_prob. The bound needs none: the loop reads it only after the body wrote it.
    sdscbundle.sdsc_execute (%zero, %cand_prob) {sdsc_filename="sdsc_0_identity.json", symbol_ids=[-1, -2]}  // groups not taken stay 0
    sdscbundle.sdsc_execute (%one, %ngroups) {sdsc_filename="sdsc_1_identity.json", symbol_ids=[-3, -4]}     // the first group always runs
    sdscbundle.sdsc_execute (%zero, %acc) {sdsc_filename="sdsc_2_identity.json", symbol_ids=[-5, -6]}

    // ---- The loop: one group of G candidates per iteration, cut where the row's mass reaches p. The body
    // writes the bound it is read from: after iteration g it holds g+2 when a row is still below p, and g+1
    // when every row reached it, which is when the loop exits.
    %bound = sdscbundle.load_value_at_address %bound_addr {granularity = 1, max_value = 1024} : index
    scf.for %g = %c0 to %bound step %c1 {
      %prob_g = affine.apply #group(%g)[%cand_prob]
      %tok_g = affine.apply #group(%g)[%cand_tok]
      // T1 topkindex k=128 over V: the next 128 tokens of each row, as ids
      sdscbundle.sdsc_execute (%probs, %tok_g) {sdsc_filename="sdsc_3_topkindex.json", symbol_ids=[-7, -8]}
      // T2 topkvalue k=128 over V: their probabilities, in the same descending order
      sdscbundle.sdsc_execute (%probs, %prob_g) {sdsc_filename="sdsc_4_topkvalue.json", symbol_ids=[-9, -10]}
      // T3 maskbyindex maskval=0, in place: the next topk finds the 128 after these
      sdscbundle.sdsc_execute (%probs, %tok_g, %probs) {sdsc_filename="sdsc_5_maskbyindex.json", symbol_ids=[-11, -12, -13]}
      // T4 cumsum over the group's 128 rows, the native op
      sdscbundle.sdsc_execute (%prob_g, %cs) {sdsc_filename="sdsc_6_cumsum.json", symbol_ids=[-14, -15]}
      // T5 before = acc + cs - cand_prob: the mass before each candidate, acc broadcast over the group
      sdscbundle.sdsc_execute (%cs, %prob_g, %before) {sdsc_filename="sdsc_7_sub.json", symbol_ids=[-16, -17, -18]}
      sdscbundle.sdsc_execute (%before, %acc, %before) {sdsc_filename="sdsc_8_add.json", symbol_ids=[-19, -20, -21]}
      // T6 greaterequal, p broadcast over the group: 1 for a candidate after the mass reached p
      sdscbundle.sdsc_execute (%before, %p, %before) {sdsc_filename="sdsc_9_greaterequal.json", symbol_ids=[-22, -23, -24]}
      // T7 where3, in place: the exact cut, 0 past it (zero broadcast over the group)
      sdscbundle.sdsc_execute (%before, %zero, %prob_g, %prob_g) {sdsc_filename="sdsc_10_where3.json", symbol_ids=[-25, -26, -27, -28]}
      // T8 add: acc += the group's sum, the last row of the cumsum
      sdscbundle.sdsc_execute (%acc, %cs_last, %acc) {sdsc_filename="sdsc_11_add.json", symbol_ids=[-29, -30, -31]}
      // T9 lesserthan: cont = acc < p
      sdscbundle.sdsc_execute (%acc, %p, %cont) {sdsc_filename="sdsc_12_lesserthan.json", symbol_ids=[-32, -33, -34]}
      // T10 ngroups = min(ngroups + cont, Gmax): a row that reached p stops counting; the min is a precaution
      sdscbundle.sdsc_execute (%ngroups, %cont, %t) {sdsc_filename="sdsc_13_add.json", symbol_ids=[-35, -36, -37]}
      sdscbundle.sdsc_execute (%t, %gmax, %ngroups) {sdsc_filename="sdsc_14_minimum.json", symbol_ids=[-38, -39, -40]}
      // T11 max over the rows: the bound, at the first position of its stick. The lanes past B, when B is
      // not a whole number of sticks, are left out with coordinate masking, or, for a symbolic B, by a mul
      // with a lane mask (1 for real rows, 0 for the others) before the max.
      // With B = 1 this SDSC goes, and the bound is read from ngroups itself.
      sdscbundle.sdsc_execute (%ngroups, %bound_addr) {sdsc_filename="sdsc_15_max.json", symbol_ids=[-41, -42]}
    }
    return
  }
}
