// SPDX-License-Identifier: MIT
//! The bridge's global allocator: the system allocator, with a per-thread cache of small blocks in front of it when the
//! environment sets NMB_TCACHE=1 (read once, on the first allocation).
//!
//! The render server's encode makes about 40 heap allocations a draw in the translator (encode_bench under callgrind), and malloc
//! and free were about 10% of the render thread's encode samples (fajita CPU profile, 2026-10-24). A block of at most 256 bytes
//! (alignment at most 16) is rounded up to one of eight size classes and taken from, or given back to, the calling thread's free
//! list of that class (intrusive, at most CAP blocks a class); a miss or an overflow goes to the system allocator with the class's
//! layout, so every block of a class has the same system layout whichever thread frees it. A thread's lists go back to the system
//! when the thread ends.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::UnsafeCell;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU8, Ordering};

const CLASSES: [usize; 8] = [16, 32, 48, 64, 96, 128, 192, 256];
const CAP: u32 = 256;
const ALIGN: usize = 16;

fn class_of(size: usize) -> Option<usize> {
    CLASSES.iter().position(|&c| size <= c)
}

fn class_layout(class: usize) -> Layout {
    // SAFETY: the class sizes are non-zero multiples of ALIGN, a power of two.
    unsafe { Layout::from_size_align_unchecked(CLASSES[class], ALIGN) }
}

struct Lists {
    heads: [*mut u8; 8],
    counts: [u32; 8],
}

struct Cache(UnsafeCell<Lists>);

impl Drop for Cache {
    fn drop(&mut self) {
        let lists = self.0.get_mut();
        for class in 0..CLASSES.len() {
            let mut at = lists.heads[class];
            while !at.is_null() {
                // SAFETY: every listed block is a live system block of the class's layout, its first word the next one.
                unsafe {
                    let next = *(at as *mut *mut u8);
                    System.dealloc(at, class_layout(class));
                    at = next;
                }
            }
            lists.heads[class] = null_mut();
            lists.counts[class] = 0;
        }
    }
}

thread_local! {
    static CACHE: Cache = const { Cache(UnsafeCell::new(Lists { heads: [null_mut(); 8], counts: [0; 8] })) };
}

/// 0 unread, 1 off, 2 on.
static MODE: AtomicU8 = AtomicU8::new(0);

fn enabled() -> bool {
    match MODE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            // SAFETY: getenv reads the environment without allocating; the name is NUL-terminated.
            let value = unsafe { libc::getenv(c"NMB_TCACHE".as_ptr()) };
            // SAFETY: a non-null getenv result is a NUL-terminated string.
            let on = !value.is_null() && unsafe { *value } == b'1' as libc::c_char;
            MODE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

pub struct BridgeAlloc;

unsafe impl GlobalAlloc for BridgeAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= ALIGN && enabled() {
            if let Some(class) = class_of(layout.size()) {
                let taken = CACHE.try_with(|cache| {
                    // SAFETY: the thread's own lists; nothing here re-enters the allocator.
                    let lists = unsafe { &mut *cache.0.get() };
                    let head = lists.heads[class];
                    if !head.is_null() {
                        // SAFETY: a listed block's first word is the next block.
                        lists.heads[class] = unsafe { *(head as *mut *mut u8) };
                        lists.counts[class] -= 1;
                    }
                    head
                });
                if let Ok(block) = taken {
                    if !block.is_null() {
                        return block;
                    }
                }
                // SAFETY: a non-zero layout.
                return unsafe { System.alloc(class_layout(class)) };
            }
        }
        // SAFETY: the caller's layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() <= ALIGN && MODE.load(Ordering::Relaxed) == 2 {
            if let Some(class) = class_of(layout.size()) {
                let kept = CACHE.try_with(|cache| {
                    // SAFETY: the thread's own lists.
                    let lists = unsafe { &mut *cache.0.get() };
                    if lists.counts[class] >= CAP {
                        return false;
                    }
                    // SAFETY: the block is at least 16 bytes and the caller's to give back.
                    unsafe { *(ptr as *mut *mut u8) = lists.heads[class] };
                    lists.heads[class] = ptr;
                    lists.counts[class] += 1;
                    true
                });
                if kept != Ok(true) {
                    // SAFETY: the block was made with its class's layout.
                    unsafe { System.dealloc(ptr, class_layout(class)) };
                }
                return;
            }
        }
        // SAFETY: the caller's block and layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= ALIGN && MODE.load(Ordering::Relaxed) == 2 {
            let (from, to) = (class_of(layout.size()), class_of(new_size));
            if from.is_some() && from == to {
                return ptr;
            }
            if from.is_some() || to.is_some() {
                // SAFETY: as GlobalAlloc's default realloc: a new block, the common bytes copied, the old one given back.
                unsafe {
                    let new = self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()));
                    if !new.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    return new;
                }
            }
        }
        // SAFETY: the caller's block and layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
