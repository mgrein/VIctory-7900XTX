//! Ranges of title memory noted for the next texture pass, bounded in
//! number without losing any byte of them.
//!
//! [`super::remapped`] and [`super::page_written`] keep the ranges the
//! runtime mapped, unmapped or wrote until a texture pass takes them. When
//! no pass runs for long (no shader that picks textures by index is bound)
//! they pile up; the queue used to be emptied at 32,768 ranges, so a pass
//! that ran later never saw those writes and the textures there stayed
//! stale. At the bound the ranges are now joined (those that touch made
//! one), and if that is not enough the nearest ones are widened over the
//! gaps between them: a pass may then look at a few slots nothing wrote,
//! never miss one something did.

/// Ranges `(start, end)` of title memory, at most `MOST` of them.
#[derive(Default)]
pub(crate) struct Noted<const MOST: usize> {
    ranges: Vec<(u64, u64)>,
}

impl<const MOST: usize> Noted<MOST> {
    pub(crate) const fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Notes `[start, end)`. Returns how many ranges were widened into a
    /// neighbour to stay within `MOST` (0 nearly always).
    pub(crate) fn insert(&mut self, start: u64, end: u64) -> usize {
        // Pages are mapped and written one after the other.
        if let Some(last) = self.ranges.last_mut()
            && last.0 <= start
            && start <= last.1
        {
            last.1 = last.1.max(end);
            return 0;
        }
        let widened = if self.ranges.len() >= MOST { self.compact() } else { 0 };
        self.ranges.push((start, end));
        widened
    }

    /// The ranges, sorted with those that touch made one. (Taken out of
    /// the lock with `std::mem::take` first: the sorting needs no lock.)
    pub(crate) fn into_ranges(self) -> Vec<(u64, u64)> {
        let mut ranges = self.ranges;
        join(&mut ranges);
        ranges
    }

    /// Joins the ranges and, while more than half of `MOST` are left,
    /// widens the ones with the smallest gaps between them over those gaps
    /// until half are. Returns how many ranges went into a neighbour by
    /// widening.
    fn compact(&mut self) -> usize {
        join(&mut self.ranges);
        let keep = (MOST / 2).max(1);
        let len = self.ranges.len();
        if len <= keep {
            return 0;
        }
        // After `join` every gap is above zero; merging the `merge`
        // smallest gaps leaves `keep` ranges.
        let merge = len - keep;
        let gaps: Vec<u64> = self.ranges.windows(2).map(|pair| pair[1].0 - pair[0].1).collect();
        let mut order = gaps.clone();
        let (_, &mut most_gap, _) = order.select_nth_unstable(merge - 1);
        let below = gaps.iter().filter(|&&gap| gap < most_gap).count();
        let mut level = merge - below;
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(keep);
        out.push(self.ranges[0]);
        for (i, &next) in self.ranges[1..].iter().enumerate() {
            let gap = gaps[i];
            let widen = gap < most_gap
                || (gap == most_gap && level > 0 && {
                    level -= 1;
                    true
                });
            if widen {
                let last = out.last_mut().expect("one range pushed first");
                last.1 = last.1.max(next.1);
            } else {
                out.push(next);
            }
        }
        self.ranges = out;
        merge
    }
}

/// Ranges of memory, those that touch made one (DexNom/VIctory 921d8d3b:
/// several threads map and write, some 5,000 ranges a frame, the same
/// textures' over and over, and each was looked up).
pub(crate) fn join(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, last| {
        if next.0 > last.1 {
            return false;
        }
        last.1 = last.1.max(next.1);
        true
    });
}

#[cfg(test)]
mod tests {
    use super::{Noted, join};

    #[test]
    fn ranges_that_touch_are_one() {
        let mut ranges = vec![(40, 50), (30, 40), (35, 40), (10, 20), (45, 60), (60, 61), (1, 1)];
        join(&mut ranges);
        assert_eq!(ranges, [(1, 1), (10, 20), (30, 61)]);
        join(&mut ranges);
        assert_eq!(ranges.len(), 3);
    }

    /// Ranges that follow each other stay one, as before.
    #[test]
    fn following_ranges_are_one() {
        let mut noted = Noted::<4>::new();
        for page in 0..100u64 {
            assert_eq!(noted.insert(page << 12, (page + 1) << 12), 0);
        }
        assert!(!noted.is_empty());
        let taken = std::mem::take(&mut noted);
        assert!(noted.is_empty());
        assert_eq!(taken.into_ranges(), [(0, 100 << 12)]);
    }

    /// At the bound no byte is lost: every range inserted lies inside one
    /// that is taken, there are never more than the bound, and the gaps
    /// widened over are the smallest.
    #[test]
    fn full_widens_instead_of_dropping() {
        const MOST: usize = 64;
        let mut noted = Noted::<MOST>::new();
        let mut inserted = Vec::new();
        let mut widened = 0;
        // Out of order, with gaps of different sizes.
        for i in 0..1000u64 {
            let start = (i * 7919) % 1000 * 0x1_0000 + (i % 3) * 0x100;
            let range = (start, start + 0x1000);
            inserted.push(range);
            widened += noted.insert(range.0, range.1);
            assert!(noted.ranges.len() <= MOST);
        }
        assert!(widened > 0);
        let taken = noted.into_ranges();
        assert!(taken.len() <= MOST);
        assert!(taken.windows(2).all(|pair| pair[0].1 < pair[1].0));
        for (start, end) in inserted {
            assert!(
                taken.iter().any(|&(a, b)| a <= start && end <= b),
                "range {start:#x}..{end:#x} lost"
            );
        }
    }

    /// The smallest gaps go first: two clusters far apart stay apart.
    #[test]
    fn nearest_ranges_are_widened_first() {
        let mut noted = Noted::<4>::new();
        for &(start, end) in &[(0, 1), (3, 4), (1000, 1001), (1002, 1003)] {
            assert_eq!(noted.insert(start, end), 0);
        }
        assert_eq!(noted.insert(5000, 5001), 2);
        assert_eq!(noted.into_ranges(), [(0, 4), (1000, 1003), (5000, 5001)]);
    }
}
